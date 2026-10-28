#!/bin/bash
#SBATCH --gpus=4
#SBATCH -C a100_40g
#SBATCH --ntasks-per-node=1

#SBATCH --exclude erc-hpc-comp[235-239],erc-hpc-comp033,erc-hpc-comp040,erc-hpc-comp036

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
            --model-name latent_compression \
            --tokenizer-name ./olmo2_tokenizer.json \
            --batch-size 60 \
            --dataset-path "./data/fineweb" \
            --num-steps 15000 \
            --train-logging-freq 100 \
            --eval-freq 1000 \
            --hidden-size 1536 \
            --depth 5 \
            --betaN 0.001 \
            --beta0 0.00001 \
            --beta-cycles 10 \
            --gradient-accumulation-steps 4 
    '
