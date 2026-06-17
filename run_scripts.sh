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

9. Rerun full eval/analysis artifacts from a saved checkpoint
CUDA_VISIBLE_DEVICES=0 python analyze_checkpoint.py --checkpoint results/<run>/checkpoints/checkpoint_epoch_0005.pt

10. Train with Adam
CUDA_VISIBLE_DEVICES=0 python train.py --training.optimizer adam

11. Train with Lion and explicit betas
CUDA_VISIBLE_DEVICES=0 python train.py --training.optimizer lion --training.betas 0.9 0.99

12. Train with SGD
CUDA_VISIBLE_DEVICES=0 python train.py --training.optimizer sgd --training.momentum 0.9

13. Train with config overrides
CUDA_VISIBLE_DEVICES=0 python train.py --training.epochs 20 --training.eval_every 1 --visualization.anim_step 2

14. Train with the recommended scale-invariant analytic decoding contract made explicit
CUDA_VISIBLE_DEVICES=0 python train.py --task.targets_type softmax --task.lstm_init_type softmax --task.decode_type analytic

Note: --task.targets_type normalized uses unit-peak independent activation targets, sigmoid/BCE loss, and sigmoid-activation weighted decoding; softmax remains the recommended analytic-decoding contract.

15. Train with the reference-project DoS place-cell target branch
CUDA_VISIBLE_DEVICES=0 python train.py --task.pc_target_family difference_of_gaussians --task.pc_target_normalization global --task.pc_surround_scale 2.0

16. Train with the reference-project true DoG place-cell target branch
CUDA_VISIBLE_DEVICES=0 python train.py --task.pc_target_family true_difference_of_gaussians --task.pc_surround_scale 2.0

17. Train with emphasized timestep-0 place-cell target loss
CUDA_VISIBLE_DEVICES=0 python train.py --training.first_pos_loss_multiplier 50

18. Train with lighter statistical analysis settings and population scale histogram/GMM plots
CUDA_VISIBLE_DEVICES=0 python train.py --analysis.num_shuffles 20 --analysis.scale_discreteness_num_shuffles 50 --analysis.max_eval_trajectories 64

19. Train with only per-unit Banino-style grid geometry analysis
CUDA_VISIBLE_DEVICES=0 python train.py --analysis.compute_grid_selectivity false --analysis.compute_grid_geometry true --analysis.compute_shuffle_significance false --analysis.compute_split_half false --analysis.compute_hd_selectivity false

20. Train with the v1 linear spatial-PCA gridness baseline diagnostics
CUDA_VISIBLE_DEVICES=0 python train.py --analysis.compute_linear_spatial_pca true --analysis.linear_spatial_pca_top_k 16 --analysis.linear_spatial_pca_num_shuffles 20 --analysis.linear_spatial_pca_plot_top_n 16

21. Train while analyzing the model-returned bottleneck without an extra eval transform
CUDA_VISIBLE_DEVICES=0 python train.py --analysis.bottleneck_activation raw

22. Train with the step learning-rate scheduler
CUDA_VISIBLE_DEVICES=0 python train.py --training.lr_scheduler step --training.lr_step_size 10 --training.lr_gamma 0.8

23. Train with infinity-norm gradient clipping across all model parameters
CUDA_VISIBLE_DEVICES=0 python train.py --training.grad_clip_mode norm --training.grad_clip_scope all --training.grad_clip_norm_type inf

24. Train with signed mean-magnitude gradient processing across all model parameters
CUDA_VISIBLE_DEVICES=0 python train.py --training.grad_processing_mode sign_mean_abs --training.grad_clip_scope all --training.grad_clip 1e-5

Monitor

25. Inspect losses, decoded metrics, and per-module gradient diagnostics in TensorBoard
tensorboard --logdir results
EOF
