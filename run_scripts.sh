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
python generate_data.py --animate --anim_num_traj 4 --anim_step 2

4. Generate a smaller dataset
python generate_data.py --output data/train_small.npz --num_samples 4000 --train_only

Train

5. Train with the default workflow
CUDA_VISIBLE_DEVICES=0 python train.py

6. Train with config overrides
CUDA_VISIBLE_DEVICES=0 python train.py --training.epochs 20 --training.eval_every 1 --visualization.anim_step 2

Monitor

7. Inspect metrics in TensorBoard
tensorboard --logdir results
EOF
