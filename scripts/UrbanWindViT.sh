export CUDA_VISIBLE_DEVICES=0,1,2,3
torchrun --nproc_per_node=4 main.py --model UrbanWindViT -t full --my_path /path/to/Dataset --score 1
