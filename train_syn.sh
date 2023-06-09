#!/bin/bash
#SBATCH --job-name=contrastive_loss
#SBATCH -o job_syn.out
#SBATCH --gpus=a100:1
#SBATCH --partition=medium
#SBATCH --time=175
#SBATCH --nodelist=xgph7


nvidia-smi
source activate experiments
python3 -u train_contrastive.py >> syn_log.txt
