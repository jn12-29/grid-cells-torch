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

6. Legacy single-file generation with explicit output paths
python generate_data.py --output data/train_small.npz --eval_output data/eval_small.npz

Train

7. Train with the default workflow (reads data/latest/train.npz and data/latest/eval.npz)
CUDA_VISIBLE_DEVICES=0 python train.py

8. Train with config overrides
CUDA_VISIBLE_DEVICES=0 python train.py --training.epochs 20 --training.eval_every 1 --visualization.anim_step 2

Monitor

9. Inspect metrics in TensorBoard
tensorboard --logdir results
EOF
