#!/bin/bash
#SBATCH --time 2-00:00:00
#SBATCH --gpus=2
#SBATCH -C h200
#SBATCH --ntasks-per-node=1

#SBATCH --exclude erc-hpc-comp[235-239]

#SBATCH --cpus-per-task=8
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

srun nvidia-smi

source ./venv/bin/activate

export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=33999


srun \
    --ntasks="$SLURM_NNODES" \
    --ntasks-per-node=1 \
    bash -c '
        torchrun \
            --nnodes="$SLURM_NNODES" \
            --nproc-per-node="$SLURM_GPUS_ON_NODE" \
            --rdzv-id="$SLURM_JOB_ID" \
            --rdzv-backend=c10d \
            --rdzv-endpoint="$MASTER_ADDR:$MASTER_PORT" \
            main.py \
            --config-path ./configs/sum_merger_fc_model_config.json \
            --tokenizer-name ./olmo2_tokenizer.json \
            --batch-size 32 \
            --dataset-path "./data/fineweb" \
            --train-logging-freq 100 \
            --betaN 0.001 \
            --beta0 0.00001 \
            --beta-cycles 10 \
            --tokens-per-step 262144 \
            --exp-name "sum_merger"
    '

