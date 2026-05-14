# Checkpoint Saving Plan

## Goal

Add save-only training checkpoints so a completed or in-progress training run can export model weights and enough experiment context for later analysis. Do not implement resume training in this change.

## Final Contract

- `training.checkpoint_every`: integer, default `0`.
- `0` or a negative value disables periodic checkpoints.
- `N > 0` saves one periodic checkpoint after every `N` completed epochs.
- `training.save_final_checkpoint`: boolean, default `true`.
- When enabled, save one final checkpoint after the training loop exits normally.
- Final checkpoint saving requires at least one completed epoch; `epochs <= 0` should not write an epoch-0 final snapshot.
- Checkpoints are written under the resolved run directory, after `timestamp_save_dir` and `run_name` are applied.
- Periodic filename: `<run_dir>/checkpoints/checkpoint_epoch_XXXX.pt`.
- Final filename: `<run_dir>/checkpoints/checkpoint_final.pt`.
- `XXXX` is the number of completed epochs, matching eval artifact semantics such as `epoch_0005`.
- No checkpoint is saved for epoch 0 unless a future change explicitly adds initial-state snapshots.
- Checkpointing is independent from `training.eval_every` and `training.eval_plot_every`.
- This plan intentionally does not add `resume_from`, best-checkpoint tracking, symlinks, or retention cleanup.

## Checkpoint Payload

Use a plain `torch.save` payload with these keys:

```python
{
    "schema_version": 1,
    "epoch": completed_epoch,
    "global_step": global_step,
    "model_state_dict": cpu_model_state_dict,
    "optimizer_state_dict": cpu_optimizer_state_dict,
    "scheduler_state_dict": cpu_scheduler_state_dict_or_none,
    "config": namespace_to_dict(cfg),
    "cell_centers": {
        "pc": [torch.as_tensor(ensemble.means, dtype=torch.float32).cpu() for ensemble in pc_ens],
        "hdc": [torch.as_tensor(ensemble.means, dtype=torch.float32).reshape(-1).cpu() for ensemble in hdc_ens],
    },
}
```

Implementation notes:

- Copy payload tensors to CPU before serialization so checkpoints load cleanly on CPU-only machines.
- Do not call `model.cpu()` on the live model during periodic saving; build CPU copies with `tensor.detach().cpu().clone()` so later epochs keep using the original device.
- Recursively copy tensors inside optimizer and scheduler state dicts to CPU for consistency.
- Include `cell_centers` because `GridCellsRNN.pc_ensembles` and `GridCellsRNN.hdc_ensembles` are plain lists, not `nn.Module` children, so their centers are not included in `model.state_dict()`.
- Include optimizer and scheduler state even though resume is not implemented yet; this is low-cost and avoids a format break if resume is added later.
- Store `cell_centers` as CPU tensors, not NumPy arrays, so default `torch.load(path, map_location="cpu")` remains compatible with PyTorch's safer loading path.
- Use `torch.save(payload, tmp_path)` followed by `os.replace(tmp_path, final_path)` to avoid leaving a corrupt target file if writing is interrupted.
- Put the temporary file in the same directory as the final checkpoint so `os.replace` is atomic on the target filesystem.
- Treat `filename` as a basename and reject path separators so checkpoints cannot escape `<cfg.training.save_dir>/checkpoints`.

## Implementation Steps

1. Update config defaults.

Change `config.yaml` under `training:`:

```yaml
  checkpoint_every: 0           # save checkpoint every N completed epochs; 0 disables periodic checkpoints
  save_final_checkpoint: true   # save checkpoints/checkpoint_final.pt after normal training completion
```

Place these near `save_dir`, `run_name`, and `timestamp_save_dir`.

2. Add CLI overrides.

Change `grid_cells/common/config_cli.py` in `CONFIG_OVERRIDE_SPECS["training"]`:

```python
"checkpoint_every": {"type": int},
"save_final_checkpoint": {"type": str2bool},
```

This keeps `python train.py --training.checkpoint_every 5 --training.save_final_checkpoint false` consistent with the existing config-backed override style.

3. Add runtime checkpoint helpers.

Change `grid_cells/training/runtime.py`:

- Add `build_checkpoint_payload(model, optimizer, lr_scheduler, cfg, pc_ens, hdc_ens, epoch, global_step)`.
- Add `save_checkpoint(model, optimizer, lr_scheduler, cfg, pc_ens, hdc_ens, epoch, global_step, filename)`.
- `filename` should be either `checkpoint_epoch_XXXX.pt` or `checkpoint_final.pt`.
- The helper should create `<cfg.training.save_dir>/checkpoints` and return the final path for logging/tests.
- Keep this in `runtime.py` rather than `session.py` so `TrainingSession` remains orchestration-only.

The callable contract should be:

```python
def save_checkpoint(
    model,
    optimizer,
    lr_scheduler,
    cfg,
    pc_ens,
    hdc_ens,
    epoch: int,
    global_step: int,
    filename: str,
) -> str:
    ...
```

4. Wire the hook.

Change `grid_cells/training/session.py`:

- Add `save_checkpoint: Callable` to `TrainingSessionHooks`.
- In `TrainingSession.run()`, pass `model`, `optimizer`, `lr_scheduler`, `cfg`, `pc_ens`, `hdc_ens`, `epoch`, `global_step`, and `filename` to the hook.
- Save periodic checkpoints after each `_run_epoch()` call returns.
- Save final checkpoint after the `for epoch in range(...)` loop completes normally.
- Skip the final checkpoint when no epoch completed, including `epochs <= 0`.
- Do not save final checkpoint from `finally`; interrupted or failed training should not be labeled final.
- Keep `writer.close()` in `finally` as it is today.
- Update all existing `TrainingSessionHooks(...)` construction sites with a `save_checkpoint` stub, including tests.

Suggested scheduling shape:

```python
completed_epoch = epoch + 1
global_step = self._run_epoch(...)
checkpoint_every = getattr(self.cfg.training, "checkpoint_every", 0)
if checkpoint_every > 0 and completed_epoch % checkpoint_every == 0:
    self.hooks.save_checkpoint(..., epoch=completed_epoch, global_step=global_step, filename=f"checkpoint_epoch_{completed_epoch:04d}.pt")

completed_epochs = self.cfg.training.epochs
if completed_epochs > 0 and getattr(self.cfg.training, "save_final_checkpoint", True):
    self.hooks.save_checkpoint(..., epoch=completed_epochs, global_step=global_step, filename="checkpoint_final.pt")
```

5. Wire the entrypoint.

Change `train.py`:

- Import `save_checkpoint` from `grid_cells.training.runtime`.
- Pass `save_checkpoint=save_checkpoint` into `TrainingSessionHooks`.
- Update test-only `TrainingSessionHooks(...)` calls in `tests/test_train_step.py` with a no-op or capturing `save_checkpoint` stub.

6. Update user-facing docs and scripts.

Change `README.md` and `README.zh.md` with the same structure and content by language:

- Add a Quick Start example: `python train.py --training.checkpoint_every 5`.
- Add `checkpoints/checkpoint_epoch_XXXX.pt` and `checkpoints/checkpoint_final.pt` to Outputs.
- Add a More Details note that checkpoints are save-only and do not resume training yet.

Change `run_scripts.sh`:

- Add a Train command for checkpointing, for example:

```bash
CUDA_VISIBLE_DEVICES=0 python train.py --training.checkpoint_every 5
```

- Keep the printed command numbering continuous after adding the new item.

## Tests

Add targeted tests to `tests/test_train_step.py`.

1. CLI override coverage.

- Extend `test_register_config_overrides_supports_shared_sections` with `--training.checkpoint_every 5` and `--training.save_final_checkpoint false`.
- Extend `test_parse_train_args_supports_task_model_training_and_visualization_overrides` with the same fields.

2. Checkpoint helper test.

- Use `tmp_path` and a tiny `torch.nn.Linear(1, 1)` or the existing small `GridCellsRNN` fixture.
- If using `torch.nn.Linear`, pass dummy `pc_ens` and `hdc_ens` objects with `.means` arrays so `cell_centers` is exercised.
- Call `save_checkpoint(...)` directly.
- Load with `torch.load(path, map_location="cpu")`.
- Assert these keys exist: `schema_version`, `epoch`, `global_step`, `model_state_dict`, `optimizer_state_dict`, `scheduler_state_dict`, `config`, `cell_centers`.
- Assert the checkpoint path is under `<tmp_path>/checkpoints/`.
- Assert checkpoint tensors are on CPU and `cell_centers` entries are tensors, not NumPy arrays.

3. Session scheduling test.

- Use the existing `TrainingSessionHooks` injection style.
- Monkeypatch `_run_epoch` to increment and return `global_step` without real data loading.
- Stub `get_cell_ensembles`, `GridCellsRNN`, `_build_scorer`, optimizer, scheduler, loaders, and evaluation as needed.
- Set `epochs=3`, `checkpoint_every=2`, `save_final_checkpoint=True`.
- Assert periodic save is called once for `checkpoint_epoch_0002.pt` with `epoch=2`.
- Assert final save is called once for `checkpoint_final.pt` with `epoch=3`.
- Add or combine a case where `checkpoint_every=0` and `save_final_checkpoint=False` produces no calls.
- Add or combine a zero-epoch case where `epochs=0` and `save_final_checkpoint=True` produces no calls.
- Add or combine an exception case where `_run_epoch` raises and no final checkpoint is written.

## Verification Commands

Run the smallest relevant checks after implementation:

```bash
pytest tests/test_train_step.py -k "checkpoint or config_overrides or parse_train_args"
python -m py_compile train.py grid_cells/training/runtime.py grid_cells/training/session.py grid_cells/common/config_cli.py
```

If README or script examples changed, also search for stale or inconsistent terms:

```bash
rg "checkpoint_every|save_final_checkpoint|checkpoint_epoch|checkpoint_final|resume" README.md README.zh.md run_scripts.sh config.yaml grid_cells tests
```

## Non-Goals

- Do not add resume training.
- Do not add best checkpoint selection.
- Do not add checkpoint retention or cleanup.
- Do not add symlinks such as `latest.pt` unless explicitly requested later.
- Do not alter evaluation artifact timing.
- Do not change dataset generation behavior.

## Open Follow-Ups

- A future resume feature should define `training.resume_from`, optimizer/scheduler restoration, `global_step` restoration, RNG restoration, and dataloader epoch semantics in a separate plan.
- A future best-checkpoint feature should first decide which metric is authoritative, for example lowest eval `pos_mse` or highest `grid_score_60 max`.
