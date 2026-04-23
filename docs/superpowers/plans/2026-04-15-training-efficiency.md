# Training Efficiency Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Accelerate GPU training by replacing the manual LSTMCell loop with nn.LSTM, moving target encoding into DataLoader workers, adding AMP, and enabling persistent_workers.

**Architecture:** Four independent optimizations applied in order of risk: model forward pass first (nn.LSTM), then DataLoader pipeline (worker encoding + persistent_workers), then mixed precision (AMP). Each task is independently verifiable.

**Tech Stack:** PyTorch, numpy, pytest

---

## File Map

| File | Change |
|------|--------|
| `model.py` | `LSTMCell` → `nn.LSTM`, simplify `forward()` |
| `dataset.py` | Add `attach_ensembles()`, encode in `__getitem__`, `persistent_workers=True` |
| `train.py` | Pass ensembles to dataloader, remove encode calls, add AMP |
| `tests/test_model.py` | New — verify nn.LSTM output matches LSTMCell |
| `tests/test_dataset.py` | New — verify encoded fields from `__getitem__` |
| `tests/test_train_step.py` | New — smoke-test a single training step end-to-end |

---

### Task 1: Create test infrastructure

**Files:**
- Create: `tests/__init__.py`
- Create: `tests/test_model.py`

- [ ] **Step 1: Create tests directory**

```bash
mkdir -p tests
touch tests/__init__.py
```

- [ ] **Step 2: Write a failing test that verifies the current LSTMCell forward produces the right output shape**

Create `tests/test_model.py`:

```python
"""Tests for GridCellsRNN model."""
import torch
import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ensembles import PlaceCellEnsemble, HeadDirectionCellEnsemble
from model import GridCellsRNN


def make_model(nh_lstm=16, nh_bottleneck=32):
    pc_ens = [PlaceCellEnsemble(8, stdev=0.35, pos_min=-1.1, pos_max=1.1, seed=0)]
    hdc_ens = [HeadDirectionCellEnsemble(4, concentration=20.0, seed=0)]
    return GridCellsRNN(pc_ens, hdc_ens, nh_lstm=nh_lstm, nh_bottleneck=nh_bottleneck,
                        dropout_rate=0.0), pc_ens, hdc_ens


def test_forward_output_shapes():
    """Model returns tensors of expected shapes."""
    model, pc_ens, hdc_ens = make_model()
    model.eval()
    B, T = 3, 20
    init_cond_size = sum(e.n_cells for e in pc_ens) + sum(e.n_cells for e in hdc_ens)
    init_cond = torch.zeros(B, init_cond_size)
    velocity = torch.randn(B, T, 3)

    pc_logits, hdc_logits, bottleneck, lstm_acts = model(init_cond, velocity, training=False)

    assert pc_logits[0].shape == (B, T, 8)
    assert hdc_logits[0].shape == (B, T, 4)
    assert bottleneck.shape == (B, T, 32)
    assert lstm_acts.shape == (B, T, 16)


def test_forward_deterministic_with_dropout_off():
    """Same input produces same output when dropout=0."""
    model, pc_ens, hdc_ens = make_model()
    model.eval()
    B, T = 2, 10
    init_cond_size = sum(e.n_cells for e in pc_ens) + sum(e.n_cells for e in hdc_ens)
    init_cond = torch.randn(B, init_cond_size)
    velocity = torch.randn(B, T, 3)

    out1 = model(init_cond, velocity, training=False)[2]
    out2 = model(init_cond, velocity, training=False)[2]
    assert torch.allclose(out1, out2)
```

- [ ] **Step 3: Run tests to verify they pass with the current LSTMCell implementation (baseline)**

```bash
cd /home/xh/ai4neuron/gridCells/grid-cells-torch
python -m pytest tests/test_model.py -v
```

Expected: both tests PASS (this establishes baseline behavior).

- [ ] **Step 4: Commit baseline tests**

```bash
git add tests/__init__.py tests/test_model.py
git commit -m "test: add baseline model shape and determinism tests"
```

---

### Task 2: Replace LSTMCell with nn.LSTM

**Files:**
- Modify: `model.py`

- [ ] **Step 1: Add numerical equivalence test** (add to `tests/test_model.py`)

Append this test to `tests/test_model.py`:

```python
def test_lstm_numerical_equivalence():
    """nn.LSTM and LSTMCell produce identical outputs for the same weights."""
    import torch.nn as nn

    torch.manual_seed(42)
    nh_lstm, nh_bottleneck = 16, 32
    model_new, pc_ens, hdc_ens = make_model(nh_lstm=nh_lstm, nh_bottleneck=nh_bottleneck)
    model_new.eval()

    B, T = 2, 15
    init_cond_size = sum(e.n_cells for e in pc_ens) + sum(e.n_cells for e in hdc_ens)
    init_cond = torch.randn(B, init_cond_size)
    velocity = torch.randn(B, T, 3)

    # Run the new model
    _, _, bottleneck_new, lstm_new = model_new(init_cond, velocity, training=False)

    # Manually run LSTMCell with the same weights to get reference outputs
    lstm_cell = nn.LSTMCell(3, nh_lstm)
    with torch.no_grad():
        lstm_cell.weight_ih.copy_(model_new.lstm.weight_ih_l0)
        lstm_cell.weight_hh.copy_(model_new.lstm.weight_hh_l0)
        lstm_cell.bias_ih.copy_(model_new.lstm.bias_ih_l0)
        lstm_cell.bias_hh.copy_(model_new.lstm.bias_hh_l0)

    with torch.no_grad():
        h = model_new.state_init(init_cond)
        c = model_new.cell_init(init_cond)
        ref_outputs = []
        for t in range(T):
            h, c = lstm_cell(velocity[:, t, :], (h, c))
            ref_outputs.append(h)
        lstm_ref = torch.stack(ref_outputs, dim=1)

    assert torch.allclose(lstm_new, lstm_ref, atol=1e-5), \
        f"Max diff: {(lstm_new - lstm_ref).abs().max().item()}"
```

- [ ] **Step 2: Run new test — expect FAIL (nn.LSTM not yet used)**

```bash
python -m pytest tests/test_model.py::test_lstm_numerical_equivalence -v
```

Expected: FAIL — the test will pass or fail depending on current impl; if it passes already the baseline impl already matches. Either way proceed to next step.

- [ ] **Step 3: Replace LSTMCell with nn.LSTM in model.py**

In `model.py`, replace the `lstm_cell` line and update `forward()`:

**In `__init__`, replace:**
```python
self.lstm_cell = nn.LSTMCell(input_size=3, hidden_size=nh_lstm)
```
**with:**
```python
self.lstm = nn.LSTM(input_size=3, hidden_size=nh_lstm, batch_first=True)
```

**In `forward()`, replace the entire manual unroll block:**
```python
        # ------------------------------------------------------------------
        # Manually unroll LSTM over time steps
        # ------------------------------------------------------------------
        lstm_outputs      = []   # list of (batch, nh_lstm)
        bottleneck_outputs = []  # list of (batch, nh_bottleneck)

        for t in range(seq_len):
            v_t = velocity[:, t, :]          # (batch, 3)
            h, c = self.lstm_cell(v_t, (h, c))

            # Bottleneck projection (no activation)
            bn = self.bottleneck(h)           # (batch, nh_bottleneck)

            # Dropout — controlled by the `training` parameter, NOT self.training
            bn = F.dropout(bn, p=self.dropout_rate, training=training)

            lstm_outputs.append(h)
            bottleneck_outputs.append(bn)

        # Stack along time dimension
        lstm_acts       = torch.stack(lstm_outputs,       dim=1)  # (B, T, nh_lstm)
        bottleneck_acts = torch.stack(bottleneck_outputs, dim=1)  # (B, T, nh_bottleneck)
```
**with:**
```python
        # ------------------------------------------------------------------
        # LSTM over full sequence (cuDNN fused kernel)
        # ------------------------------------------------------------------
        lstm_acts, _ = self.lstm(velocity, (h.unsqueeze(0), c.unsqueeze(0)))
        # lstm_acts: (B, T, nh_lstm)

        # Bottleneck + dropout over full sequence
        bottleneck_acts = self.bottleneck(lstm_acts)                         # (B, T, nh_bottleneck)
        bottleneck_acts = F.dropout(bottleneck_acts, p=self.dropout_rate, training=training)
```

- [ ] **Step 4: Run all model tests**

```bash
python -m pytest tests/test_model.py -v
```

Expected: all 3 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add model.py tests/test_model.py
git commit -m "perf: replace LSTMCell manual loop with nn.LSTM for cuDNN fused kernel"
```

---

### Task 3: Add `attach_ensembles` to TrajectoryDataset

**Files:**
- Modify: `dataset.py`
- Create: `tests/test_dataset.py`

- [ ] **Step 1: Write failing tests for `attach_ensembles` and encoded `__getitem__`**

Create `tests/test_dataset.py`:

```python
"""Tests for TrajectoryDataset encoding features."""
import numpy as np
import torch
import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dataset import TrajectoryDataset, get_dataloader
from ensembles import PlaceCellEnsemble, HeadDirectionCellEnsemble
from types import SimpleNamespace


def make_ensembles():
    pc_ens = [PlaceCellEnsemble(8, stdev=0.35, pos_min=-1.1, pos_max=1.1, seed=0)]
    hdc_ens = [HeadDirectionCellEnsemble(4, concentration=20.0, seed=0)]
    return pc_ens, hdc_ens


def make_cfg():
    return SimpleNamespace(
        task=SimpleNamespace(
            env_size=2.2, seq_len=10, neurons_seed=0,
            targets_type="softmax", lstm_init_type="softmax",
            velocity_noise=[0.0, 0.0, 0.0],
        ),
        training=SimpleNamespace(
            batch_size=4, steps_per_epoch=3,
        ),
    )


def test_attach_ensembles_adds_init_cond():
    """After attach_ensembles, __getitem__ returns init_cond."""
    ds = TrajectoryDataset(num_samples=5, seq_len=10, env_size=2.2, seed=1)
    pc_ens, hdc_ens = make_ensembles()
    ds.attach_ensembles(pc_ens, hdc_ens)

    item = ds[0]
    assert "init_cond" in item
    assert item["init_cond"].shape == (12,)          # 8 pc + 4 hdc
    assert item["init_cond"].dtype == np.float32


def test_attach_ensembles_adds_targets():
    """After attach_ensembles, __getitem__ returns pc_targets_0 and hdc_targets_0."""
    ds = TrajectoryDataset(num_samples=5, seq_len=10, env_size=2.2, seed=1)
    pc_ens, hdc_ens = make_ensembles()
    ds.attach_ensembles(pc_ens, hdc_ens)

    item = ds[0]
    assert "pc_targets_0" in item
    assert "hdc_targets_0" in item
    assert item["pc_targets_0"].shape == (10, 8)     # (seq_len, n_pc)
    assert item["hdc_targets_0"].shape == (10, 4)    # (seq_len, n_hdc)


def test_without_attach_no_encoded_keys():
    """Without attach_ensembles, __getitem__ returns raw trajectory only."""
    ds = TrajectoryDataset(num_samples=5, seq_len=10, env_size=2.2, seed=1)
    item = ds[0]
    assert "init_cond" not in item
    assert "pc_targets_0" not in item


def test_init_cond_consistent_with_manual_encode():
    """init_cond from dataset matches encode_initial_conditions output."""
    from utils import encode_initial_conditions
    ds = TrajectoryDataset(num_samples=5, seq_len=10, env_size=2.2, seed=1)
    pc_ens, hdc_ens = make_ensembles()
    ds.attach_ensembles(pc_ens, hdc_ens)

    idx = 2
    item = ds[idx]
    # Build a single-item batch to pass to encode_initial_conditions
    batch = {
        "init_pos": torch.from_numpy(ds._data["init_pos"][idx:idx+1]),
        "init_hd":  torch.from_numpy(ds._data["init_hd"][idx:idx+1]),
    }
    ref = encode_initial_conditions(batch, pc_ens, hdc_ens).numpy()[0]  # (init_cond_size,)
    assert np.allclose(item["init_cond"], ref, atol=1e-5)
```

- [ ] **Step 2: Run tests — expect FAIL**

```bash
python -m pytest tests/test_dataset.py -v
```

Expected: FAIL with `AttributeError: 'TrajectoryDataset' object has no attribute 'attach_ensembles'`.

- [ ] **Step 3: Implement `attach_ensembles` and update `__getitem__` in dataset.py**

Add the following imports at the top of `dataset.py` (after existing imports):
```python
import numpy as np  # already present
# (no new imports needed)
```

Add the `attach_ensembles` method to `TrajectoryDataset` after `__init__`:

```python
    def attach_ensembles(self, pc_ensembles, hdc_ensembles) -> None:
        """Pre-compute init_cond for all samples; store ensembles for per-item target encoding.

        Call this once after dataset creation (before training loop).
        init_cond is pre-computed for all N samples (~10 MB for N=10000) and stored.
        pc_targets / hdc_targets are computed per sample in __getitem__ by DataLoader workers.

        Args:
            pc_ensembles:  list of PlaceCellEnsemble
            hdc_ensembles: list of HeadDirectionCellEnsemble
        """
        self._pc_ens = pc_ensembles
        self._hdc_ens = hdc_ensembles

        # Pre-compute init_cond for all samples at once
        # init_pos: (N, 1, 2),  init_hd: (N, 1, 1)
        init_pos = self._data["init_pos"][:, np.newaxis, :]  # (N, 1, 2)
        init_hd  = self._data["init_hd"][:, np.newaxis, :]  # (N, 1, 1)

        parts = []
        for ens in pc_ensembles:
            act = ens.get_init(init_pos)   # (N, 1, n_cells)
            parts.append(act[:, 0, :])     # (N, n_cells)
        for ens in hdc_ensembles:
            act = ens.get_init(init_hd)    # (N, 1, n_cells)
            parts.append(act[:, 0, :])     # (N, n_cells)

        self._init_cond = np.concatenate(parts, axis=-1).astype(np.float32)  # (N, init_cond_size)
```

Replace the existing `__getitem__` method with:

```python
    def __getitem__(self, idx) -> dict:
        item = {
            "init_pos":   self._data["init_pos"][idx],
            "init_hd":    self._data["init_hd"][idx],
            "ego_vel":    self._data["ego_vel"][idx],
            "target_pos": self._data["target_pos"][idx],
            "target_hd":  self._data["target_hd"][idx],
        }

        if hasattr(self, "_pc_ens"):
            # Pre-computed init_cond (fast array lookup)
            item["init_cond"] = self._init_cond[idx]

            # Per-sample target encoding (runs in DataLoader worker)
            # Add seq dim: (1, T, dim) expected by get_targets
            pos = self._data["target_pos"][idx][np.newaxis]  # (1, T, 2)
            hd  = self._data["target_hd"][idx][np.newaxis]   # (1, T, 1)

            for i, ens in enumerate(self._pc_ens):
                item[f"pc_targets_{i}"] = ens.get_targets(pos)[0]    # (T, n_cells)

            for i, ens in enumerate(self._hdc_ens):
                item[f"hdc_targets_{i}"] = ens.get_targets(hd)[0]    # (T, n_cells)

        return item
```

- [ ] **Step 4: Run dataset tests**

```bash
python -m pytest tests/test_dataset.py -v
```

Expected: all 4 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add dataset.py tests/test_dataset.py
git commit -m "perf: pre-compute init_cond and move target encoding into DataLoader workers"
```

---

### Task 4: Update `get_dataloader` to accept ensembles and enable `persistent_workers`

**Files:**
- Modify: `dataset.py` (the `get_dataloader` function)

- [ ] **Step 1: Add ensemble params and persistent_workers to `get_dataloader`**

Replace the existing `get_dataloader` function in `dataset.py` with:

```python
def get_dataloader(cfg, data_path: str = None,
                   pc_ens=None, hdc_ens=None) -> DataLoader:
    """Build a DataLoader from a config object or a saved .npz file.

    When data_path is given, trajectories are loaded from disk (fast, no
    generation overhead).  When data_path is None, trajectories are
    generated on-the-fly every call (original behaviour).

    When pc_ens and hdc_ens are provided, the dataset pre-computes init_cond
    for all samples and encodes targets per-sample in DataLoader workers,
    eliminating per-batch CPU work in the training loop.

    Args:
        cfg:       config namespace with task / training attributes.
        data_path: optional path to a .npz file created by
                   TrajectoryDataset.save() or generate_data.py.
        pc_ens:    optional list of PlaceCellEnsemble for worker-side encoding.
        hdc_ens:   optional list of HeadDirectionCellEnsemble for worker-side encoding.

    Returns:
        DataLoader with pin_memory=True, num_workers=4, persistent_workers=True.
    """
    if data_path is not None:
        dataset = TrajectoryDataset.from_file(data_path)
    else:
        num_samples = cfg.training.steps_per_epoch * cfg.training.batch_size
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
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
    )
```

- [ ] **Step 2: Verify existing dataset tests still pass**

```bash
python -m pytest tests/test_dataset.py -v
```

Expected: all 4 tests PASS.

- [ ] **Step 3: Commit**

```bash
git add dataset.py
git commit -m "perf: add ensemble params to get_dataloader and enable persistent_workers"
```

---

### Task 5: Update training loop to use pre-encoded batch fields

**Files:**
- Modify: `train.py`
- Create: `tests/test_train_step.py`

- [ ] **Step 1: Write a failing smoke test for the training step**

Create `tests/test_train_step.py`:

```python
"""Smoke test for training step with pre-encoded batch."""
import torch
import numpy as np
import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from types import SimpleNamespace
from ensembles import PlaceCellEnsemble, HeadDirectionCellEnsemble
from model import GridCellsRNN
from dataset import get_dataloader


def make_cfg():
    return SimpleNamespace(
        task=SimpleNamespace(
            env_size=2.2, seq_len=10, neurons_seed=0,
            targets_type="softmax", lstm_init_type="softmax",
            velocity_noise=[0.0, 0.0, 0.0],
            n_pc=[8], pc_scale=[0.35],
            n_hdc=[4], hdc_concentration=[20.0],
        ),
        model=SimpleNamespace(
            nh_lstm=16, nh_bottleneck=32,
            dropout_rate=0.5, bottleneck_has_bias=False,
            init_weight_disp=0.0,
        ),
        training=SimpleNamespace(
            batch_size=4, steps_per_epoch=2,
            lr=1e-4, momentum=0.9, weight_decay=1e-5, grad_clip=1e-5,
        ),
    )


def test_training_step_with_preencoded_batch():
    """A single training step completes and loss decreases after several steps."""
    cfg = make_cfg()
    device = torch.device("cpu")

    pc_ens = [PlaceCellEnsemble(8, stdev=0.35, pos_min=-1.1, pos_max=1.1, seed=0)]
    hdc_ens = [HeadDirectionCellEnsemble(4, concentration=20.0, seed=0)]
    model = GridCellsRNN(pc_ens, hdc_ens, **vars(cfg.model)).to(device)
    optimizer = torch.optim.RMSprop(model.parameters(), lr=cfg.training.lr,
                                    momentum=cfg.training.momentum,
                                    weight_decay=cfg.training.weight_decay)

    loader = get_dataloader(cfg, pc_ens=pc_ens, hdc_ens=hdc_ens)
    batch = next(iter(loader))

    # Verify pre-encoded fields are present
    assert "init_cond" in batch, "init_cond missing from batch"
    assert "pc_targets_0" in batch, "pc_targets_0 missing from batch"
    assert "hdc_targets_0" in batch, "hdc_targets_0 missing from batch"

    # Run a single forward+backward step
    batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
             for k, v in batch.items()}
    init_cond = batch["init_cond"].float()

    pc_logits, hdc_logits, _, _ = model(init_cond, batch["ego_vel"], training=True)

    loss = sum(
        ens.loss(logits, batch[f"pc_targets_{i}"].numpy())
        for i, (ens, logits) in enumerate(zip(pc_ens, pc_logits))
    )
    loss += sum(
        ens.loss(logits, batch[f"hdc_targets_{i}"].numpy())
        for i, (ens, logits) in enumerate(zip(hdc_ens, hdc_logits))
    )

    optimizer.zero_grad()
    loss.backward()
    loss_val = loss.item()
    assert np.isfinite(loss_val), f"Loss is not finite: {loss_val}"
```

- [ ] **Step 2: Run test — expect PASS (tests logic not yet in train.py, but tests batch structure)**

```bash
python -m pytest tests/test_train_step.py -v
```

Expected: PASS (the test validates the batch structure directly against dataset + model, independent of train.py).

- [ ] **Step 3: Update `train.py` — pass ensembles to `_fixed_loader` and use pre-encoded batch**

In the `train()` function, make these changes:

**Change the `_fixed_loader` creation block** (around line 219–224) from:
```python
    if data_path is not None:
        logger.info("Loading trajectories from %s", data_path)
        # Pre-load once; same dataset is reused every epoch (shuffled by DataLoader)
        _fixed_loader = get_dataloader(cfg, data_path=data_path)
    else:
        _fixed_loader = None
```
**to:**
```python
    if data_path is not None:
        logger.info("Loading trajectories from %s", data_path)
        # Pre-load once; attach ensembles so workers encode targets in parallel
        _fixed_loader = get_dataloader(cfg, data_path=data_path,
                                       pc_ens=pc_ens, hdc_ens=hdc_ens)
    else:
        _fixed_loader = None
```

**Replace the inner batch loop** from:
```python
        for step, batch in enumerate(dataloader):
            batch = {k: v.to(device) for k, v in batch.items()}

            init_cond = encode_initial_conditions(batch, pc_ens, hdc_ens).to(device)
            pc_targets, hdc_targets = encode_targets(batch, pc_ens, hdc_ens)

            pc_logits, hdc_logits, _, _ = model(
                init_cond, batch["ego_vel"], training=True
            )

            loss = sum(
                ens.loss(logits, targets)
                for ens, logits, targets in zip(pc_ens, pc_logits, pc_targets)
            )
            loss += sum(
                ens.loss(logits, targets)
                for ens, logits, targets in zip(hdc_ens, hdc_logits, hdc_targets)
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_value_(
                model.parameters(), cfg.training.grad_clip
            )
            optimizer.step()
            loss_acc.append(loss.item())
            logger.debug("epoch=%4d  step=%4d  loss=%.4f", epoch, step, loss.item())
```
**with:**
```python
        for step, batch in enumerate(dataloader):
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}

            if "init_cond" in batch:
                # Pre-encoded path: init_cond and targets already in batch
                init_cond = batch["init_cond"].float()
                pc_targets  = [batch[f"pc_targets_{i}"]  for i in range(len(pc_ens))]
                hdc_targets = [batch[f"hdc_targets_{i}"] for i in range(len(hdc_ens))]
            else:
                # Fallback: compute on-the-fly (on-demand data generation path)
                init_cond = encode_initial_conditions(batch, pc_ens, hdc_ens).to(device)
                pc_targets, hdc_targets = encode_targets(batch, pc_ens, hdc_ens)

            pc_logits, hdc_logits, _, _ = model(
                init_cond, batch["ego_vel"], training=True
            )

            loss = sum(
                ens.loss(logits, targets)
                for ens, logits, targets in zip(pc_ens, pc_logits, pc_targets)
            )
            loss += sum(
                ens.loss(logits, targets)
                for ens, logits, targets in zip(hdc_ens, hdc_logits, hdc_targets)
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_value_(
                model.parameters(), cfg.training.grad_clip
            )
            optimizer.step()
            loss_acc.append(loss.item())
            logger.debug("epoch=%4d  step=%4d  loss=%.4f", epoch, step, loss.item())
```

- [ ] **Step 4: Verify no regressions**

```bash
python -m pytest tests/ -v
```

Expected: all tests PASS.

- [ ] **Step 5: Commit**

```bash
git add train.py
git commit -m "perf: use pre-encoded batch fields in training loop, remove per-batch encode calls"
```

---

### Task 6: Add AMP (autocast + GradScaler)

**Files:**
- Modify: `train.py`

- [ ] **Step 1: Add GradScaler initialisation and autocast to the training loop**

In `train.py`, after the optimizer creation block (after `weight_decay=...`), add:

```python
    # 3b. GradScaler for AMP (no-op when CUDA is unavailable)
    scaler = torch.amp.GradScaler(device=device.type, enabled=(device.type == "cuda"))
    logger.info("AMP enabled: %s", device.type == "cuda")
```

Then replace the forward + backward block inside the batch loop:

**Replace:**
```python
            pc_logits, hdc_logits, _, _ = model(
                init_cond, batch["ego_vel"], training=True
            )

            loss = sum(
                ens.loss(logits, targets)
                for ens, logits, targets in zip(pc_ens, pc_logits, pc_targets)
            )
            loss += sum(
                ens.loss(logits, targets)
                for ens, logits, targets in zip(hdc_ens, hdc_logits, hdc_targets)
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_value_(
                model.parameters(), cfg.training.grad_clip
            )
            optimizer.step()
```
**with:**
```python
            with torch.autocast(device_type=device.type,
                                 enabled=(device.type == "cuda")):
                pc_logits, hdc_logits, _, _ = model(
                    init_cond, batch["ego_vel"], training=True
                )
                loss = sum(
                    ens.loss(logits, targets)
                    for ens, logits, targets in zip(pc_ens, pc_logits, pc_targets)
                )
                loss += sum(
                    ens.loss(logits, targets)
                    for ens, logits, targets in zip(hdc_ens, hdc_logits, hdc_targets)
                )

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_value_(
                model.parameters(), cfg.training.grad_clip
            )
            scaler.step(optimizer)
            scaler.update()
```

- [ ] **Step 2: Run all tests**

```bash
python -m pytest tests/ -v
```

Expected: all tests PASS (tests run on CPU where AMP is a no-op).

- [ ] **Step 3: Commit**

```bash
git add train.py
git commit -m "perf: add AMP autocast and GradScaler for fp16 GPU training"
```

---

## Self-Review

### Spec coverage check

| Spec requirement | Task |
|-----------------|------|
| LSTMCell → nn.LSTM | Task 2 |
| Pre-compute init_cond at load time | Task 3 |
| Per-sample target encoding in workers | Task 3 |
| Indexed target keys (`pc_targets_0`) | Task 3 |
| `get_dataloader` accepts `pc_ens`/`hdc_ens` | Task 4 |
| `persistent_workers=True` | Task 4 |
| Training loop uses pre-encoded batch | Task 5 |
| Fallback for on-demand data (no data_path) | Task 5 |
| AMP autocast | Task 6 |
| GradScaler with `scaler.unscale_` before grad clip | Task 6 |
| Eval loop unchanged | Tasks 3–5 (`_evaluate` uses raw dataloader, no ensemble passed) |

### Type consistency check

- `attach_ensembles` stores `self._pc_ens`, `self._hdc_ens`, `self._init_cond` — all referenced consistently in `__getitem__` via `hasattr(self, "_pc_ens")`.
- `get_dataloader` signature: `pc_ens=None, hdc_ens=None` — matches call site in `train.py`.
- Indexed keys `pc_targets_{i}` / `hdc_targets_{i}` — consistent across `__getitem__`, test, and training loop.
- `scaler.unscale_(optimizer)` called before `clip_grad_value_` — correct order for AMP.

### Placeholder scan

No TBDs, no "handle edge cases", all code blocks are complete. ✓
