#!/bin/bash
#SBATCH --job-name=grey2col
#SBATCH --output=/scratch/jl13122/MAE_SIMP/outputs/output_%j.txt
#SBATCH --error=/scratch/jl13122/MAE_SIMP/errors/error_%j.txt
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH -t 5-00:00:00
#SBATCH --mem=64GB
#SBATCH --gres=gpu
#SBATCH --cpus-per-task=16

singularity exec --nv --bind /scratch/$USER --overlay /scratch/$USER/overlay_gpt2.ext3:rw /scratch/$USER/cuda11.4.2-cudnn8.2.4-devel-ubuntu20.04.3.sif bash -c '

source /ext3/env.sh
conda activate MAE_env
cd classWvit
python train_gpt.py
'


