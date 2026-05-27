#!/usr/bin/env bash
set -euo pipefail

cat <<'EOF'
grid-cells-torch command cheat sheet

Data

1. Generate the default train/eval datasets
python generate_data.py

2. Generate datasets with preview PDF and MP4
python generate_data.py --visualize --animate

3. Generate datasets with custom animation sampling
python generate_data.py --animate --visualization.anim_num_traj 4 --visualization.anim_step 2

4. Generate a smaller dataset directory
python generate_data.py --data_generation.num_samples 4000 --train_only

5. Generate datasets with config-backed defaults overridden from the CLI
python generate_data.py --task.seq_len 800 --data_generation.num_workers 4 --visualization.anim_fps 30

6. File-mode generation with explicit output paths
python generate_data.py --output data/train_small.npz --eval_output data/eval_small.npz

Train

7. Train with the default workflow (reads data/latest/train.npz, eval.npz, and cells.npz)
CUDA_VISIBLE_DEVICES=0 python train.py

8. Train with periodic checkpoint saving
CUDA_VISIBLE_DEVICES=0 python train.py --training.checkpoint_every 5

9. Train with SGD
CUDA_VISIBLE_DEVICES=0 python train.py --training.optimizer sgd --training.momentum 0.9

10. Train with config overrides
CUDA_VISIBLE_DEVICES=0 python train.py --training.epochs 20 --training.eval_every 1 --visualization.anim_step 2

11. Train with the recommended scale-invariant analytic decoding contract made explicit
CUDA_VISIBLE_DEVICES=0 python train.py --task.targets_type softmax --task.lstm_init_type softmax --task.decode_type analytic

Note: --task.targets_type normalized uses unit-peak independent activation targets, sigmoid/BCE loss, and sigmoid-activation weighted decoding; softmax remains the recommended analytic-decoding contract.

12. Train with emphasized timestep-0 place-cell target loss
CUDA_VISIBLE_DEVICES=0 python train.py --training.first_pos_loss_multiplier 50

13. Train with lighter statistical analysis settings and population scale histogram/GMM plots
CUDA_VISIBLE_DEVICES=0 python train.py --analysis.num_shuffles 20 --analysis.scale_discreteness_num_shuffles 50 --analysis.max_eval_trajectories 64

14. Train with only per-unit Banino-style grid geometry analysis
CUDA_VISIBLE_DEVICES=0 python train.py --analysis.compute_grid_selectivity false --analysis.compute_grid_geometry true --analysis.compute_shuffle_significance false --analysis.compute_split_half false --analysis.compute_hd_selectivity false

15. Train with the default cosine learning-rate scheduler made explicit
CUDA_VISIBLE_DEVICES=0 python train.py --training.lr_scheduler cosine --training.lr_min 1e-5

Monitor

16. Inspect metrics in TensorBoard
tensorboard --logdir results
EOF
