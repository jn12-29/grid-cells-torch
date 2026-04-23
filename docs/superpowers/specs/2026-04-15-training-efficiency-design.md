# Training Efficiency Design

**Date:** 2026-04-15  
**Scope:** GPU training with pre-generated data (`--data_path`)

---

## Problem

Four bottlenecks identified in the current training loop:

1. **Manual LSTMCell loop** — `model.py` unrolls `seq_len=1000` steps in a Python for-loop via `LSTMCell`. Prevents cuDNN fused kernels.
2. **Per-batch target encoding** — `encode_initial_conditions` / `encode_targets` are called in the main process every batch, doing GPU→CPU numpy round-trips (softmax over B×T×256 per batch).
3. **No mixed precision** — all compute runs in fp32.
4. **Worker respawn overhead** — DataLoader workers are recreated each epoch.

---

## Design

### 1. Replace `LSTMCell` loop with `nn.LSTM` (model.py)

- Replace `self.lstm_cell = nn.LSTMCell(3, nh_lstm)` with `self.lstm = nn.LSTM(3, nh_lstm, batch_first=True)`.
- In `forward()`, delete the `for t in range(seq_len)` loop. Replace with a single call:
  ```python
  lstm_out, _ = self.lstm(velocity, (h.unsqueeze(0), c.unsqueeze(0)))
  # lstm_out: (B, T, nh_lstm)
  ```
- Apply bottleneck and dropout over the full sequence in one shot:
  ```python
  bottleneck_acts = self.bottleneck(lstm_out)              # (B, T, nh_bottleneck)
  bottleneck_acts = F.dropout(bottleneck_acts, p=self.dropout_rate, training=training)
  ```
- Numerical equivalence: `nn.LSTM` and `LSTMCell` share identical weight layout (`weight_ih`, `weight_hh`, `bias_ih`, `bias_hh`). Results are bit-identical on CPU; on GPU with cuDNN may differ by floating-point rounding at fp32, which is acceptable.

### 2. Move target encoding into DataLoader workers (dataset.py + train.py)

**Goal:** eliminate main-process CPU work per batch.

**`init_cond` pre-computation (one-time at load):**
- Add `attach_ensembles(pc_ens, hdc_ens)` method to `TrajectoryDataset`.
- On call, pre-compute `self._init_cond: np.ndarray` of shape `(N, init_cond_size)` for all N samples using the existing `encode_initial_conditions` logic (runs once, ~10 MB for N=10000, init_cond_size=268).
- Store ensembles as `self._pc_ens`, `self._hdc_ens`.

**Per-sample target computation in `__getitem__`:**
- When ensembles are attached, `__getitem__` additionally computes and returns:
  - `init_cond`: pre-stored slice `self._init_cond[idx]` → shape `(init_cond_size,)` float32
  - `pc_targets`: list of arrays `(T, n_pc_i)` float32 — computed by `ens.get_targets` on the single trajectory
  - `hdc_targets`: list of arrays `(T, n_hdc_i)` float32
- When ensembles are not attached, `__getitem__` returns only the raw trajectory dict (existing behavior, used by eval).
- DataLoader's default collate stacks these into `(B, ...)` tensors automatically.

**`get_dataloader` signature change:**
```python
def get_dataloader(cfg, data_path=None, pc_ens=None, hdc_ens=None) -> DataLoader
```
When `pc_ens` is provided, calls `dataset.attach_ensembles(pc_ens, hdc_ens)` before returning.

Add `persistent_workers=True` to all DataLoader calls (avoids worker respawn each epoch).

**Training loop (train.py):**
- Pass `pc_ens` / `hdc_ens` when building `_fixed_loader`.
- Replace `encode_initial_conditions` / `encode_targets` calls with direct reads from the batch dict.
- Move target tensors to device in the batch loop alongside other batch keys.
- Eval `get_dataloader` call is unchanged (no ensembles passed → old behavior).

**Collate note:** `pc_targets` / `hdc_targets` are lists of arrays per sample. The default collate does not handle per-item lists of arrays. Use a custom `collate_fn` that stacks each ensemble's targets into a single tensor: `torch.stack([item["pc_targets"][i] for item in batch])` per ensemble index. Alternatively, flatten the list into indexed keys (`pc_targets_0`, `pc_targets_1`, ...) which collate handles natively.
- **Decision:** use indexed keys (`pc_targets_0`, `hdc_targets_0`) to keep the collate simple.

### 3. AMP with autocast + GradScaler (train.py)

Only enabled when `device.type == "cuda"`.

```python
scaler = torch.amp.GradScaler(enabled=(device.type == "cuda"))

# in training loop:
with torch.autocast(device_type=device.type, enabled=(device.type == "cuda")):
    pc_logits, hdc_logits, _, _ = model(init_cond, batch["ego_vel"], training=True)
    loss = compute_loss(...)

optimizer.zero_grad()
scaler.scale(loss).backward()
torch.nn.utils.clip_grad_value_(model.parameters(), cfg.training.grad_clip)
scaler.step(optimizer)
scaler.update()
```

`ens.loss()` uses `torch.as_tensor(targets, dtype=predictions.dtype, ...)` — this correctly picks up the fp16 dtype under autocast.

### 4. `persistent_workers=True` (dataset.py)

Add to `DataLoader(...)` call. Workers stay alive across epochs, eliminating fork overhead (~0.5–1 s per epoch with num_workers=4).

---

## Files Changed

| File | Change |
|------|--------|
| `model.py` | `LSTMCell` → `nn.LSTM`; simplify `forward()` |
| `dataset.py` | `attach_ensembles()`, `__getitem__` encodes when attached, `persistent_workers=True`, indexed target keys |
| `train.py` | Pass ensembles to `get_dataloader`; remove encode calls; add AMP |
| `utils.py` | No changes (encode functions kept for eval) |

---

## What Is NOT Changed

- Eval loop (`_evaluate`) continues to use `encode_initial_conditions` / `encode_targets` as before.
- `generate_data.py`, `scores.py`, `ensembles.py` — untouched.
- Config / CLI interface — untouched.
