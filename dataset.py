"""Generate, load, and batch synthetic trajectory datasets.

The dataset matches the trajectory format expected by the grid-cell training
loop and can optionally attach ensemble-derived initial conditions and targets
so workers hand back ready-to-train batches.

Usage:
    ds = TrajectoryDataset(num_samples=100_000, seq_len=100, env_size=2.2)
    ds.save("data/train.npz")
    ds = TrajectoryDataset.from_file("data/train.npz")
    loader = get_dataloader(cfg, pc_ens=pc_ensembles, hdc_ens=hdc_ensembles)
"""

import json

import numpy as np
from torch.utils.data import Dataset, DataLoader

from encoding import EnsembleEncoder
from trajectory_generation import TrajectoryGenerator


class TrajectoryDataset(Dataset):
    """
    Generates and stores random-walk trajectories in a square environment.

    Each trajectory simulates rodent locomotion using an Ornstein-Uhlenbeck
    process for angular velocity and truncated Gaussian for translational speed.

    Args:
        num_samples:     Number of trajectories to pre-generate.
        seq_len:         Number of time steps per trajectory.
        env_size:        Side length of the square environment in metres.
                         Positions are bounded to [-env_size/2, env_size/2].
        velocity_noise:  Tuple/list (σx, σy, σω) of Gaussian noise std devs
                         added to ego_vel components.  Pass (0,0,0) to disable.
        seed:            Optional integer seed for reproducibility.
    """

    # Motion model constants (Sargolini 2006 / Milford 2010 rodent motion model)
    _DT = TrajectoryGenerator._DT   # time step, seconds
    _B = TrajectoryGenerator._B     # angular velocity decay coefficient (OU process)
    _MV = TrajectoryGenerator._MV   # mean translational speed  (m/s, ~10 cm/s)
    _SV = TrajectoryGenerator._SV   # std  translational speed   (m/s)
    _MW = TrajectoryGenerator._MW   # mean angular velocity      (rad/s)
    _SW = TrajectoryGenerator._SW   # std angular velocity ~94 deg/s (~1.63 rad/s)

    def __init__(
        self,
        num_samples: int,
        seq_len: int,
        env_size: float,
        velocity_noise=(0.0, 0.0, 0.0),
        seed=None,
        num_workers: int = 1,
        chunk_size: int = None,
        progress_callback=None,
    ):
        self.num_samples = num_samples
        self.seq_len = seq_len
        self.env_size = env_size
        self.velocity_noise = tuple(velocity_noise)
        self._generator = TrajectoryGenerator(
            seq_len=self.seq_len,
            env_size=self.env_size,
            velocity_noise=self.velocity_noise,
        )

        self._data = self._generate_all(
            base_seed=seed,
            num_workers=num_workers,
            chunk_size=chunk_size,
            progress_callback=progress_callback,
        )

    def attach_ensembles(self, pc_ensembles, hdc_ensembles) -> None:
        """Attach ensembles and pre-compute initial-condition encodings.

        The full initial-condition tensor is cheap to store once for all
        samples. Target encodings stay per-sample so DataLoader workers can
        compute them in parallel inside ``__getitem__``.
        """
        self._pc_ens = pc_ensembles
        self._hdc_ens = hdc_ensembles
        self._encoder = EnsembleEncoder(pc_ensembles, hdc_ensembles)
        self._init_cond = self._encoder.encode_initial_conditions(
            self._data["init_pos"],
            self._data["init_hd"],
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _generate_trajectory(self, rng: np.random.Generator):
        """Generate a single trajectory; returns dict of float32 arrays."""
        return self._generator.generate_trajectory(rng)

    def _generate_all(
        self,
        base_seed,
        num_workers: int = 1,
        chunk_size: int = None,
        progress_callback=None,
    ) -> dict:
        """Pre-generate all trajectories and stack them into arrays."""
        return self._generator.generate_many(
            num_samples=self.num_samples,
            base_seed=base_seed,
            num_workers=num_workers,
            chunk_size=chunk_size,
            progress_callback=progress_callback,
        )

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx) -> dict:
        item = {
            "init_pos":   self._data["init_pos"][idx],
            "init_hd":    self._data["init_hd"][idx],
            "ego_vel":    self._data["ego_vel"][idx],
            "target_pos": self._data["target_pos"][idx],
            "target_hd":  self._data["target_hd"][idx],
        }

        if hasattr(self, "_encoder"):
            item["init_cond"] = self._init_cond[idx]

            pos = self._data["target_pos"][idx][np.newaxis, :]
            hd = self._data["target_hd"][idx][np.newaxis, :]
            encoded_targets = self._encoder.encode_targets(pos, hd)

            for i, pc_targets in enumerate(encoded_targets.pc_targets):
                item[f"pc_targets_{i}"] = pc_targets[0].astype(np.float32)

            for i, hdc_targets in enumerate(encoded_targets.hdc_targets):
                item[f"hdc_targets_{i}"] = hdc_targets[0].astype(np.float32)

        return item

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Save all pre-generated trajectories to a compressed .npz file.

        Metadata (env_size, seq_len, velocity_noise) is stored as a JSON
        string in the 'meta' field so the dataset can be reconstructed
        exactly from the file.

        Args:
            path: destination file path, e.g. "data/trajectories.npz".
                  Parent directories are created automatically.
        """
        import os
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        meta = json.dumps({
            "env_size":      self.env_size,
            "seq_len":       self.seq_len,
            "velocity_noise": list(self.velocity_noise),
            "num_samples":   self.num_samples,
        })
        np.savez_compressed(
            path,
            init_pos=self._data["init_pos"],
            init_hd=self._data["init_hd"],
            ego_vel=self._data["ego_vel"],
            target_pos=self._data["target_pos"],
            target_hd=self._data["target_hd"],
            meta=np.array(meta),          # stored as 0-d string array
        )
        print(f"Saved {self.num_samples} trajectories to {path}")

    @classmethod
    def from_file(cls, path: str) -> "TrajectoryDataset":
        """Load a dataset that was previously saved with save().

        Args:
            path: path to a .npz file created by save().

        Returns:
            A TrajectoryDataset instance backed by the loaded arrays.
            No new trajectories are generated.
        """
        data = np.load(path, allow_pickle=False)
        meta = json.loads(str(data["meta"]))

        # Build a shell instance without running __init__ generation
        obj = cls.__new__(cls)
        obj.num_samples    = meta["num_samples"]
        obj.seq_len        = meta["seq_len"]
        obj.env_size       = meta["env_size"]
        obj.velocity_noise = tuple(meta["velocity_noise"])
        obj._generator = TrajectoryGenerator(
            seq_len=obj.seq_len,
            env_size=obj.env_size,
            velocity_noise=obj.velocity_noise,
        )
        obj._data = {
            "init_pos":   data["init_pos"],
            "init_hd":    data["init_hd"],
            "ego_vel":    data["ego_vel"],
            "target_pos": data["target_pos"],
            "target_hd":  data["target_hd"],
        }
        print(f"Loaded {obj.num_samples} trajectories from {path}")
        return obj


# ---------------------------------------------------------------------------
# DataLoader factory
# ---------------------------------------------------------------------------

def get_dataloader(
    cfg,
    data_path: str = None,
    pc_ens=None,
    hdc_ens=None,
    num_samples: int = None,
    shuffle: bool = True,
    batch_size: int = None,
) -> DataLoader:
    """Build a DataLoader from a config object or a saved .npz file.

    When data_path is given, trajectories are loaded from disk (fast, no
    generation overhead).  When data_path is None, trajectories are
    generated on-the-fly every call (original behaviour).

    When ensembles are attached, initial conditions are pre-computed once and
    targets are encoded per-sample inside DataLoader workers.

    Args:
        cfg:       config namespace with task / training attributes.
        data_path: optional path to a .npz file created by
                   TrajectoryDataset.save() or generate_data.py.
        pc_ens:    optional list of place-cell ensembles.
        hdc_ens:   optional list of head-direction-cell ensembles.
        num_samples: optional sample count override for generated datasets.
        shuffle:   whether to shuffle the dataset each epoch.

    Returns:
        DataLoader with pin_memory=True and persistent workers enabled.
    """
    if data_path is not None:
        dataset = TrajectoryDataset.from_file(data_path)
    else:
        num_samples = num_samples or (
            cfg.training.steps_per_epoch * cfg.training.batch_size
        )
        dataset = TrajectoryDataset(
            num_samples=num_samples,
            seq_len=cfg.task.seq_len,
            env_size=cfg.task.env_size,
            velocity_noise=cfg.task.velocity_noise,
            seed=cfg.task.neurons_seed,
        )

    if pc_ens is not None and hdc_ens is not None:
        dataset.attach_ensembles(pc_ens, hdc_ens)

    num_workers = 4
    resolved_batch_size = batch_size if batch_size is not None else cfg.training.batch_size
    return DataLoader(
        dataset,
        batch_size=resolved_batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
    )
