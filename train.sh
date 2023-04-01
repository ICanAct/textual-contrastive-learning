#!/bin/bash
#SBATCH --job-name=contrastive_loss
#SBATCH -o job_contrastive.out
#SBATCH --gpus=a100:1
#SBATCH --partition=medium
#SBATCH --time=150
#SBATCH --nodelist=xgph6

nvidia-smi
source activate experiments
python3 -u train_contrastive.py >> contrastive_log.txt
