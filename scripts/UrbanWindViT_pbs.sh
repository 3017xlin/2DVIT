#!/bin/bash
#PBS -N UrbanWindViT
#PBS -l select=1:ncpus=16:ngpus=4:mem=128gb
#PBS -l walltime=72:00:00
#PBS -q gpu_as
#PBS -P gs_mae_hongying.li
#PBS -j oe
#PBS -o urbanwindvit.log

# Move into the directory the job was submitted from.
cd "$PBS_O_WORKDIR"

# Activate the conda env that lives on the project volume.
source /usr/local/anaconda2025/etc/profile.d/conda.sh
conda activate /projects_vol/gp_hongying.li/yiheng/envs/airfrans

# Four ranks, one per visible GPU. world_size is auto-detected by torchrun
# and read inside train.py via dist.get_world_size().
export CUDA_VISIBLE_DEVICES=0,1,2,3
export PYTHONUNBUFFERED=1
torchrun --nproc_per_node=4 main.py \
    --model UrbanWindViT \
    -t full \
    --my_path /projects_vol/gp_hongying.li/yiheng/Dataset \
    --score 0
