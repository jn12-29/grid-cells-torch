"""Synthetic trajectory generation helpers for grid-cell training datasets.

This module separates random-walk generation from the PyTorch ``Dataset``
wrapper so generation logic can evolve independently from loading and encoding.

Usage:
    generator = TrajectoryGenerator(seq_len=100, env_size=2.2)
    data = generator.generate_many(num_samples=1024, base_seed=0)
"""

import math
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np


def _generate_chunk_worker(task) -> tuple[int, dict]:
    """Generate one contiguous chunk of trajectories in a worker process."""
    seq_len, env_size, velocity_noise, start_idx, sample_seeds = task
    generator = TrajectoryGenerator(
        seq_len=seq_len,
        env_size=env_size,
        velocity_noise=velocity_noise,
    )
    chunk_size = len(sample_seeds)
    chunk = generator.allocate_storage(chunk_size, seq_len)
    for offset, sample_seed in enumerate(sample_seeds):
        traj = generator.generate_trajectory(np.random.default_rng(sample_seed))
        chunk["init_pos"][offset] = traj["init_pos"]
        chunk["init_hd"][offset] = traj["init_hd"]
        chunk["ego_vel"][offset] = traj["ego_vel"]
        chunk["target_pos"][offset] = traj["target_pos"]
        chunk["target_hd"][offset] = traj["target_hd"]

    return start_idx, chunk


class TrajectoryGenerator:
    """Generate random-walk trajectories inside a square environment."""

    _DT = 0.02
    _B = 0.26
    _MV = 0.1
    _SV = 0.13
    _MW = 0.0
    _SW = 0.52 * np.pi

    def __init__(self, seq_len: int, env_size: float, velocity_noise=(0.0, 0.0, 0.0)):
        self.seq_len = seq_len
        self.env_size = env_size
        self.velocity_noise = tuple(velocity_noise)

    @staticmethod
    def allocate_storage(num_samples: int, seq_len: int) -> dict:
        """Allocate output arrays for a dataset or chunk."""
        return {
            "init_pos": np.empty((num_samples, 2), dtype=np.float32),
            "init_hd": np.empty((num_samples, 1), dtype=np.float32),
            "ego_vel": np.empty((num_samples, seq_len, 3), dtype=np.float32),
            "target_pos": np.empty((num_samples, seq_len, 2), dtype=np.float32),
            "target_hd": np.empty((num_samples, seq_len, 1), dtype=np.float32),
        }

    @staticmethod
    def build_sample_seeds(base_seed, num_samples: int) -> np.ndarray:
        """Build deterministic per-sample seeds so worker count does not change data."""
        seed_sequence = np.random.SeedSequence(base_seed)
        return seed_sequence.generate_state(num_samples * 2, dtype=np.uint32).reshape(
            num_samples,
            2,
        )

    def generate_trajectory(self, rng: np.random.Generator) -> dict:
        """Generate a single trajectory and return float32 arrays."""
        total_steps = self.seq_len
        dt = self._DT
        decay = self._B
        half = self.env_size / 2.0

        init_pos = rng.uniform(-half, half, size=(2,)).astype(np.float32)
        init_hd = rng.uniform(-np.pi, np.pi, size=(1,)).astype(np.float32)

        pos = init_pos.copy()
        hd = float(init_hd[0])
        angular_velocity = 0.0
        ego_vel = np.empty((total_steps, 3), dtype=np.float32)
        target_pos = np.empty((total_steps, 2), dtype=np.float32)
        target_hd = np.empty((total_steps, 1), dtype=np.float32)

        for step in range(total_steps):
            angular_velocity = (1.0 - decay) * angular_velocity + decay * rng.normal(
                self._MW,
                self._SW,
            )

            speed = rng.normal(self._MV, self._SV)
            while speed <= 0.0:
                speed = rng.normal(self._MV, self._SV)

            hd = hd + angular_velocity * dt
            hd = (hd + np.pi) % (2.0 * np.pi) - np.pi

            dx = speed * np.cos(hd) * dt
            dy = speed * np.sin(hd) * dt
            new_x = pos[0] + dx
            new_y = pos[1] + dy

            ego_vel[step, 0] = speed * np.cos(hd)
            ego_vel[step, 1] = speed * np.sin(hd)
            ego_vel[step, 2] = angular_velocity

            if new_x < -half or new_x > half:
                hd = np.pi - hd
                hd = (hd + np.pi) % (2.0 * np.pi) - np.pi
                new_x = np.clip(new_x, -half, half)
            if new_y < -half or new_y > half:
                hd = -hd
                hd = (hd + np.pi) % (2.0 * np.pi) - np.pi
                new_y = np.clip(new_y, -half, half)

            pos[0] = new_x
            pos[1] = new_y
            target_pos[step] = pos
            target_hd[step, 0] = hd

        noise_x, noise_y, noise_w = self.velocity_noise
        if noise_x != 0.0:
            ego_vel[:, 0] += rng.normal(0.0, noise_x, size=total_steps).astype(np.float32)
        if noise_y != 0.0:
            ego_vel[:, 1] += rng.normal(0.0, noise_y, size=total_steps).astype(np.float32)
        if noise_w != 0.0:
            ego_vel[:, 2] += rng.normal(0.0, noise_w, size=total_steps).astype(np.float32)

        return {
            "init_pos": init_pos,
            "init_hd": init_hd,
            "ego_vel": ego_vel,
            "target_pos": target_pos,
            "target_hd": target_hd,
        }

    def generate_many(
        self,
        num_samples: int,
        base_seed,
        num_workers: int = 1,
        chunk_size: int = None,
        progress_callback=None,
    ) -> dict:
        """Pre-generate all trajectories and stack them into arrays."""
        num_workers = max(1, int(num_workers))
        num_workers = min(num_workers, max(1, num_samples))

        if chunk_size is None:
            chunk_size = max(1, min(2048, math.ceil(num_samples / max(num_workers * 8, 1))))
        else:
            chunk_size = max(1, int(chunk_size))

        storage = self.allocate_storage(num_samples, self.seq_len)
        sample_seeds = self.build_sample_seeds(base_seed, num_samples)
        tasks = []
        for start_idx in range(0, num_samples, chunk_size):
            end_idx = min(start_idx + chunk_size, num_samples)
            tasks.append(
                (
                    self.seq_len,
                    self.env_size,
                    self.velocity_noise,
                    start_idx,
                    sample_seeds[start_idx:end_idx],
                )
            )

        completed_samples = 0

        def commit_chunk(chunk_index: int, start_idx: int, chunk: dict) -> None:
            nonlocal completed_samples
            chunk_count = len(chunk["init_pos"])
            end_idx = start_idx + chunk_count
            for key, value in chunk.items():
                storage[key][start_idx:end_idx] = value

            completed_samples += chunk_count
            if progress_callback is not None:
                progress_callback(
                    {
                        "chunk_index": chunk_index,
                        "num_chunks": len(tasks),
                        "chunk_start": start_idx,
                        "chunk_end": end_idx,
                        "completed_samples": completed_samples,
                        "total_samples": num_samples,
                        "chunk_data": chunk,
                    }
                )

        if num_workers == 1 or len(tasks) == 1:
            for chunk_index, task in enumerate(tasks):
                start_idx, chunk = _generate_chunk_worker(task)
                commit_chunk(chunk_index, start_idx, chunk)
            return storage

        future_to_chunk = {}
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            for chunk_index, task in enumerate(tasks):
                future = executor.submit(_generate_chunk_worker, task)
                future_to_chunk[future] = chunk_index

            for future in as_completed(future_to_chunk):
                chunk_index = future_to_chunk[future]
                start_idx, chunk = future.result()
                commit_chunk(chunk_index, start_idx, chunk)

        return storage
