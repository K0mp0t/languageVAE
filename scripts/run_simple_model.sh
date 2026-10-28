#!/bin/bash
#SBATCH --cpus-per-task=4
#SBATCH -p interruptible_gpu,gpu,nmes_gpu
#SBATCH --gres=gpu:1
#SBATCH -C rtx6000
#SBATCH --mem=16384
#SBATCH --output=/scratch/users/%u/job_outputs/%j.out
#SBATCH --chdir=/scratch/users/k25137033/dlms/languageVAE

echo "Job started on node: $(hostname)"

nvidia-smi

echo "Current working directory: $(pwd)"

source ./venv/bin/activate

pip --version

python main.py
