"""
Trajectory dataset for grid cell navigation experiments.

Generates synthetic random-walk trajectories mimicking rodent motion in a
square environment (Sargolini et al. 2006 motion model), matching the format
of the original TFRecord dataset used in Banino et al. Nature 2018.

Workflow
--------
Generate once and save:
    ds = TrajectoryDataset(num_samples=100_000, seq_len=100, env_size=2.2)
    ds.save("data/trajectories.npz")

Load for training:
    ds = TrajectoryDataset.from_file("data/trajectories.npz")
    loader = DataLoader(ds, batch_size=10, shuffle=True)
"""

import json
import numpy as np
from torch.utils.data import Dataset, DataLoader


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
    _DT = 0.02   # time step, seconds
    _B  = 0.26   # angular velocity decay coefficient (OU process)
    _MV = 0.1    # mean translational speed  (m/s, ~10 cm/s)
    _SV = 0.13   # std  translational speed   (m/s)
    _MW = 0.0    # mean angular velocity      (rad/s)
    _SW = 0.52 * np.pi  # std angular velocity ~94 deg/s (~1.63 rad/s)

    def __init__(
        self,
        num_samples: int,
        seq_len: int,
        env_size: float,
        velocity_noise=(0.0, 0.0, 0.0),
        seed=None,
    ):
        self.num_samples = num_samples
        self.seq_len = seq_len
        self.env_size = env_size
        self.velocity_noise = tuple(velocity_noise)

        rng = np.random.default_rng(seed)
        self._data = self._generate_all(rng)

    def attach_ensembles(self, pc_ensembles, hdc_ensembles) -> None:
        """Attach ensembles and pre-compute initial-condition encodings.

        The full initial-condition tensor is cheap to store once for all
        samples. Target encodings stay per-sample so DataLoader workers can
        compute them in parallel inside ``__getitem__``.
        """
        self._pc_ens = pc_ensembles
        self._hdc_ens = hdc_ensembles

        init_pos = self._data["init_pos"][:, np.newaxis, :]
        init_hd = self._data["init_hd"][:, np.newaxis, :]

        parts = []
        for ens in pc_ensembles:
            parts.append(ens.get_init(init_pos)[:, 0, :])
        for ens in hdc_ensembles:
            parts.append(ens.get_init(init_hd)[:, 0, :])

        self._init_cond = np.concatenate(parts, axis=-1).astype(np.float32)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _generate_trajectory(self, rng: np.random.Generator):
        """Generate a single trajectory; returns dict of float32 arrays."""
        T   = self.seq_len
        dt  = self._DT
        B   = self._B
        half = self.env_size / 2.0

        # ----- initial state -----
        init_pos = rng.uniform(-half, half, size=(2,)).astype(np.float32)
        init_hd  = rng.uniform(-np.pi, np.pi, size=(1,)).astype(np.float32)

        pos = init_pos.copy()
        hd  = float(init_hd[0])
        w   = 0.0  # angular velocity (rad/s), initialised to zero

        # output buffers
        ego_vel    = np.empty((T, 3), dtype=np.float32)
        target_pos = np.empty((T, 2), dtype=np.float32)
        target_hd  = np.empty((T, 1), dtype=np.float32)

        for t in range(T):
            # 1. Update angular velocity (Ornstein-Uhlenbeck)
            w = (1.0 - B) * w + B * rng.normal(self._MW, self._SW)

            # 2. Update translational speed (truncated Gaussian, positive only)
            v = rng.normal(self._MV, self._SV)
            while v <= 0.0:
                v = rng.normal(self._MV, self._SV)

            # 3. Update heading
            hd = hd + w * dt
            # wrap to [-pi, pi]
            hd = (hd + np.pi) % (2.0 * np.pi) - np.pi

            # 4. Update position
            dx = v * np.cos(hd) * dt
            dy = v * np.sin(hd) * dt
            new_x = pos[0] + dx
            new_y = pos[1] + dy

            # 5. Record ego_vel BEFORE boundary reflection, so that
            #    ego_vel[t] * dt matches (target_pos[t] - target_pos[t-1])
            #    on non-reflection steps.
            ego_vel[t, 0] = v * np.cos(hd)   # global-frame vx (pre-reflection)
            ego_vel[t, 1] = v * np.sin(hd)   # global-frame vy (pre-reflection)
            ego_vel[t, 2] = w                 # angular velocity

            # 6. Boundary reflection (modifies pos and hd, but not ego_vel)
            if new_x < -half or new_x > half:
                hd = np.pi - hd          # reflect heading across y-axis
                hd = (hd + np.pi) % (2.0 * np.pi) - np.pi
                new_x = np.clip(new_x, -half, half)
            if new_y < -half or new_y > half:
                hd = -hd                 # reflect heading across x-axis
                hd = (hd + np.pi) % (2.0 * np.pi) - np.pi
                new_y = np.clip(new_y, -half, half)

            pos[0] = new_x
            pos[1] = new_y

            # 7. Record post-reflection state
            target_pos[t] = pos
            target_hd[t, 0] = hd

        # Optional velocity noise
        sx, sy, sw = self.velocity_noise
        if sx != 0.0:
            ego_vel[:, 0] += rng.normal(0.0, sx, size=T).astype(np.float32)
        if sy != 0.0:
            ego_vel[:, 1] += rng.normal(0.0, sy, size=T).astype(np.float32)
        if sw != 0.0:
            ego_vel[:, 2] += rng.normal(0.0, sw, size=T).astype(np.float32)

        return {
            "init_pos":   init_pos,
            "init_hd":    init_hd,
            "ego_vel":    ego_vel,
            "target_pos": target_pos,
            "target_hd":  target_hd,
        }

    def _generate_all(self, rng: np.random.Generator) -> dict:
        """Pre-generate all trajectories and stack them into arrays."""
        T   = self.seq_len
        N   = self.num_samples

        init_pos   = np.empty((N, 2),    dtype=np.float32)
        init_hd    = np.empty((N, 1),    dtype=np.float32)
        ego_vel    = np.empty((N, T, 3), dtype=np.float32)
        target_pos = np.empty((N, T, 2), dtype=np.float32)
        target_hd  = np.empty((N, T, 1), dtype=np.float32)

        for i in range(N):
            traj = self._generate_trajectory(rng)
            init_pos[i]   = traj["init_pos"]
            init_hd[i]    = traj["init_hd"]
            ego_vel[i]    = traj["ego_vel"]
            target_pos[i] = traj["target_pos"]
            target_hd[i]  = traj["target_hd"]

        return {
            "init_pos":   init_pos,
            "init_hd":    init_hd,
            "ego_vel":    ego_vel,
            "target_pos": target_pos,
            "target_hd":  target_hd,
        }

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

        if hasattr(self, "_pc_ens"):
            item["init_cond"] = self._init_cond[idx]

            pos = self._data["target_pos"][idx][np.newaxis, :]
            hd = self._data["target_hd"][idx][np.newaxis, :]

            for i, ens in enumerate(self._pc_ens):
                item[f"pc_targets_{i}"] = ens.get_targets(pos)[0].astype(np.float32)

            for i, ens in enumerate(self._hdc_ens):
                item[f"hdc_targets_{i}"] = ens.get_targets(hd)[0].astype(np.float32)

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
    return DataLoader(
        dataset,
        batch_size=cfg.training.batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
    )
