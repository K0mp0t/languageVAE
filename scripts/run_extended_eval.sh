#!/bin/bash
#SBATCH --gpus=1
#SBATCH --time 1:00:00
#SBATCH -C a30
#SBATCH --ntasks-per-node=1

#SBATCH --exclude erc-hpc-comp[235-239]

#SBATCH --cpus-per-task=4
#SBATCH -p he_gpu,gpu,nmes_gpu
#SBATCH --mem=16384
#SBATCH --output=/scratch/users/%u/job_outputs/%j.out
#SBATCH --chdir=/scratch/users/k25137033/dlms/languageVAE

set -e

srun -N "$SLURM_NNODES" -n "$SLURM_NNODES" bash -c '
    echo "host=$(hostname)"
    echo "SLURM_GPUS_ON_NODE=$SLURM_GPUS_ON_NODE"
    echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
'
nvidia-smi

source ./venv/bin/activate


# srun python extended_eval.py ./experiments/09_09_16_35 ./olmo2_tokenizer.json
srun python extended_eval.py ./experiments/10_21_19_52 ./olmo2_tokenizer.json
