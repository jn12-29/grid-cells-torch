CUDA_VISIBLE_DEVICES=1 python train.py --training.optimizer rmsprop --training.momentum 0.9 --training.lr 1e-5 --training.grad_clip 1e-5
CUDA_VISIBLE_DEVICES=7 python train.py --training.optimizer adamw --training.lr 1e-3 --training.grad_clip 1e-5 --training.grad_clip_mode "value" --training.first_pos_loss_multiplier 10

CUDA_VISIBLE_DEVICES=0 python train.py --training.optimizer adamw --training.lr 1e-3 --training.grad_clip 1e-5 --training.first_pos_loss_multiplier 50

CUDA_VISIBLE_DEVICES=0 python train.py --training.optimizer adam --training.lr 1e-3 --training.grad_clip 1e-5 --training.first_pos_loss_multiplier 100
