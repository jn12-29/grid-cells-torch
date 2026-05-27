CUDA_VISIBLE_DEVICES=1 python train.py --training.optimizer rmsprop --training.momentum 0.9 --training.lr 1e-5 --training.grad_clip 1e-5
CUDA_VISIBLE_DEVICES=0 python train.py --training.optimizer adamw --training.lr 1e-4 --training.grad_clip 1e-5

CUDA_VISIBLE_DEVICES=0 python train.py --training.optimizer adamw --training.lr 1e-3 --training.grad_clip 1e-5
