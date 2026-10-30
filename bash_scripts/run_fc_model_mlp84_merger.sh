#!/bin/bash

torchrun \
    --nnodes=1 \
    --nproc-per-node=gpu \
    --rdzv-backend=c10d \
    main.py \
    --config-path ./configs/mlp_merger_84_fc_model_config.json \
    --tokenizer-name ./olmo2_tokenizer.json \
    --batch-size 56 \
    --dataset-path "./data/fineweb" \
    --train-logging-freq 100 \
    --betaN 0.001 \
    --beta0 0.00001 \
    --beta-cycles 10 \
    --tokens-per-step 262144 \
    --exp-name "mlp_merger84"

