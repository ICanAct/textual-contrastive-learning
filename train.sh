#!/bin/bash
#SBATCH --job-name=contrastive_loss
#SBATCH -o job_con_sub.out
#SBATCH --gpus=a100:1
#SBATCH --partition=medium
#SBATCH --time=175
#SBATCH --nodelist=xgph8


nvidia-smi
source activate experiments
python3 -u train_contrastive.py >> con_sub_log.txt
