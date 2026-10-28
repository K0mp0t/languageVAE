import os
import time
from datetime import datetime
import argparse
import json
from contextlib import nullcontext
import math
from datetime import timedelta
import hashlib

import torch
from torch import nn
import torch.distributed as dist 
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP 

from tokenizers import Tokenizer
from datasets import load_dataset

import matplotlib.pyplot as plt
import numpy as np
from torchinfo import summary

from models import *
from beta_schedule import LinearCyclicBetaScheduler
from distribution_evaluation import get_p_mus_and_logvars, get_q_mus_and_logvars, calculate_aup
from custom_datasets import *

import random

torch.manual_seed(189833)
random.seed(189833)
np.random.seed(189833)


def reduce_and_get_value(tensor_to_reduce, n=1):
    tensor_sum = tensor_to_reduce.detach()
    n = torch.tensor(n, device=tensor_to_reduce.device)

    dist.all_reduce(tensor_sum, op=dist.ReduceOp.SUM)
    dist.all_reduce(n, op=dist.ReduceOp.SUM)

    global_value = tensor_sum / n

    return global_value.item()


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--config-path', type=str, default=None)
    parser.add_argument('--exp-name', type=str, default=None)

    parser.add_argument('--seq-len', type=int, default=256)
    parser.add_argument('--batch-size', type=int, default=64)
    
    parser.add_argument('--dataset-path', type=str, default='./data/lambada')

    parser.add_argument('--lr0', type=float, default=0.0001)
    parser.add_argument('--lrN', type=float, default=0.00001)
    parser.add_argument('--num-steps', type=int, default=-1, help='-1 for auto training horizon')
    parser.add_argument('--num-eval-steps', type=int, default=100)
    parser.add_argument('--tokens-per-step', type=int, default=1, help='-1 for auto effective batch size estimation')

    parser.add_argument('--train-logging-freq', type=int, default=100)
    parser.add_argument('--checkpointing-freq', type=int, default=1000)
    parser.add_argument('--eval-freq', type=int, default=1000)
    parser.add_argument('--resume-from', type=str, default=None)
    parser.add_argument('--resume-from-step', type=int, default=0)

    parser.add_argument('--tokenizer-name', type=str, default='bert-base-uncased')
    parser.add_argument('--beta0', type=float, default=0.00001)
    parser.add_argument('--betaN', type=float, default=0.001)
    parser.add_argument('--beta-cycles', type=float, default=10)
    parser.add_argument('--mask-ratio', type=float, default=0.15)

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    
    CONFIG_PATH = args.config_path
    DATASET_PATH = args.dataset_path

    SEQ_LEN = args.seq_len
    BATCH_SIZE = args.batch_size
    TOKENS_PER_STEP = args.tokens_per_step
    LR_0 = args.lr0
    LR_N = args.lrN
    NUM_STEPS = args.num_steps
    BETA_0 = args.beta0
    BETA_N = args.betaN
    BETA_CYCLES = args.beta_cycles
    MASK_RATIO = args.mask_ratio
    
    TOKENIZER_NAME = args.tokenizer_name

    TRAIN_LOGGING_FREQ = args.train_logging_freq
    CHECKPOINTING_FREQ = args.checkpointing_freq
    NUM_EVAL_STEPS = args.num_eval_steps
    EVAL_FREQ = args.eval_freq

    RESUME_FROM = args.resume_from
    RESUME_FROM_STEP = args.resume_from_step

    assert CONFIG_PATH is not None or RESUME_FROM is not None, 'Either config-path or resume-from must not be None'

    if RESUME_FROM is not None:
        EXP_NAME = os.path.basename(os.path.dirname(RESUME_FROM))
    elif args.exp_name is None:
        EXP_NAME = datetime.now().strftime("%m_%d_%H_%M")
    else:
        EXP_NAME = args.exp_name
    EXP_OUTPUT_DIR = os.path.join('./experiments', EXP_NAME)

    local_rank = int(os.environ["LOCAL_RANK"])

    acc = torch.accelerator.current_accelerator()
    backend = dist.get_default_backend_for_device(acc)
    dist.init_process_group(backend, device_id=local_rank)

    world_size = dist.get_world_size()
    rank = dist.get_rank()
    local_rank = dist.get_node_local_rank()

    torch.accelerator.set_device_index(local_rank)
    print(f"Starting with rank {rank}. World size: {world_size}")
    DEVICE = local_rank

    if os.path.exists(TOKENIZER_NAME):
        tokenizer = Tokenizer.from_file(TOKENIZER_NAME)
    else:
        tokenizer = Tokenizer.from_pretrained(TOKENIZER_NAME)
    tokenizer.add_special_tokens(["<|mask|>"])

    if rank == 0:
        print('Experiment settings:')
        for k, v in vars(args).items():
            print(f'  {k}: {v}')
   
    if RESUME_FROM is None:
        model_config = CustomTextVAEConfig.from_json_file(CONFIG_PATH)
        model = CustomTextVAEModel(model_config)
    else:
        if rank == 0:
            print(f'Loading model weights from {RESUME_FROM}. Resuming training from step {RESUME_FROM_STEP}')
        model = CustomTextVAEModel.from_pretrained(RESUME_FROM)

    model.to(DEVICE)
    if rank == 0:
        summary(model, input_size=(BATCH_SIZE, SEQ_LEN), dtypes=[torch.long])
    
    dist.barrier()

    ddp_model = DDP(model, device_ids=[DEVICE])

    if os.path.exists(DATASET_PATH):
        if any(fn.endswith('.parquet') for fn in os.listdir(DATASET_PATH)):
            train_loader = DDPStreamingParquetDataLoader(tokenizer, DATASET_PATH, SEQ_LEN, batch_size=BATCH_SIZE, 
                mask_ratio=MASK_RATIO, split='train', ddp_rank=rank, ddp_world_size=world_size, resume_from_step=RESUME_FROM_STEP * TOKENS_PER_STEP)
            val_loader = DDPStreamingParquetDataLoader(tokenizer, DATASET_PATH, SEQ_LEN, batch_size=BATCH_SIZE, 
                split='validation', ddp_rank=rank, ddp_world_size=world_size)
        else:
            train_loader = DDPStreamingParquetDataLoader(tokenizer, os.path.join(DATASET_PATH, 'train'), SEQ_LEN, batch_size=BATCH_SIZE, 
                mask_ratio=MASK_RATIO, split='all', ddp_rank=rank, ddp_world_size=world_size, resume_from_step=RESUME_FROM_STEP * TOKENS_PER_STEP)
            val_loader = DDPStreamingParquetDataLoader(tokenizer, os.path.join(DATASET_PATH, 'validation'), SEQ_LEN, batch_size=BATCH_SIZE, 
                split='all', ddp_rank=rank, ddp_world_size=world_size)
    else:
        splits = load_dataset(DATASET_PATH).train_test_split(test_size=0.05)
        train_dataset = splits["train"]
        val_dataset = splits["test"]

        train_sampler = ResumeDistributedSampler(train_dataset, num_replicas=world_size, rank=rank, 
            shuffle=True, skip_samples=RESUME_FROM_STEP * BATCH_SIZE)
        val_sampler = ResumeDistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False)

        train_loader = DataLoader(train_dataset, BATCH_SIZE, num_workers=4, drop_last=True, sampler=train_sampler)
        val_loader = DataLoader(val_dataset, BATCH_SIZE, num_workers=4, drop_last=True, sampler=val_sampler)

    train_loader_iterator = iter(train_loader)
    val_loader_iterator = iter(val_loader)

    rec_criterion = nn.CrossEntropyLoss(reduction='none')

    # lr_scale_ratio = math.sqrt(world_size * GRADIENT_ACCUMULATION_STEPS) / 2
    lr_scale_ratio = 1
    LR_0_SCALED = LR_0 * lr_scale_ratio
    LR_N_SCALED = LR_N * lr_scale_ratio

    if rank == 0 and lr_scale_ratio != 1:
        print(f'lr scaled by {round(lr_scale_ratio, 4)}:')
        print(f'  lr0: {round(LR_0, 6)} -> {round(LR_0_SCALED, 6)}')
        print(f'  lrN: {round(LR_N, 6)} -> {round(LR_N_SCALED, 6)}')
    
    muon_params = [p for n, p in ddp_model.named_parameters() if p.ndim == 2 and ('encoder_blocks' in n or 'decoder_blocks' in n)]
    other_params = [p for n, p in ddp_model.named_parameters() if p.ndim != 2 or ('encoder_blocks' not in n and 'decoder_blocks' not in n)]
    
    muon_optimizer = torch.optim.Muon(muon_params, lr=LR_0_SCALED, adjust_lr_fn='match_rms_adamw')
    other_optimizer = torch.optim.AdamW(other_params, lr=LR_0_SCALED)

    warmup_iters = min(int(NUM_STEPS * 0.03), 200)
    regular_iters = NUM_STEPS - warmup_iters
    
    muon_warmup_scheduler = torch.optim.lr_scheduler.LinearLR(muon_optimizer, 0.1, 1, total_iters=warmup_iters)
    muon_regular_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(muon_optimizer, T_max=regular_iters, eta_min=LR_N_SCALED)
    muon_lr_scheduler = torch.optim.lr_scheduler.SequentialLR(
        muon_optimizer, [muon_warmup_scheduler, muon_regular_scheduler], [warmup_iters]
    )

    other_warmup_scheduler = torch.optim.lr_scheduler.LinearLR(other_optimizer, 0.1, 1, total_iters=warmup_iters)
    other_regular_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(other_optimizer, T_max=regular_iters, eta_min=LR_N_SCALED)
    other_lr_scheduler = torch.optim.lr_scheduler.SequentialLR(
        other_optimizer, [other_warmup_scheduler, other_regular_scheduler], [warmup_iters]
    )
    
    if RESUME_FROM is not None:
        resume_from_dir = os.path.dirname(RESUME_FROM)

        muon_optimizer.load_state_dict(torch.load(os.path.join(resume_from_dir, 'muon_optimizer.pth'), map_location='cpu'))
        muon_lr_scheduler.load_state_dict(torch.load(os.path.join(resume_from_dir, 'muon_lr_scheduler.pth'), map_location='cpu'))

        other_optimizer.load_state_dict(torch.load(os.path.join(resume_from_dir, 'other_optimizer.pth'), map_location='cpu'))
        other_lr_scheduler.load_state_dict(torch.load(os.path.join(resume_from_dir, 'other_lr_scheduler.pth'), map_location='cpu'))

    total_params = sum(p.numel() for p in ddp_model.parameters())
    trainable_params = sum(
        p.numel() for name, p in ddp_model.named_parameters()
        if p.requires_grad
    )

    num_muon_params = sum(p.numel() for p in muon_params)
    num_other_params = sum(p.numel() for p in other_params)

    assert num_muon_params + num_other_params == total_params

    target_tokens = 12 * total_params

    if TOKENS_PER_STEP < 1:
        target_batch_tokens = 2 ** 18 * (target_tokens / 1_321_210_944) ** 0.383
        TOKENS_PER_STEP = target_batch_tokens

    no_accumulation_batch_tokens = BATCH_SIZE * SEQ_LEN * world_size
    GRADIENT_ACCUMULATION_STEPS = TOKENS_PER_STEP // no_accumulation_batch_tokens + (TOKENS_PER_STEP % no_accumulation_batch_tokens != 0)

    if NUM_STEPS < 1:
        NUM_STEPS = target_tokens // (TOKENS_PER_STEP) + 1

    beta_scheduler = LinearCyclicBetaScheduler(NUM_STEPS, BETA_CYCLES, BETA_0, BETA_N)

    if rank == 0:
        print(f"Total params:   {total_params:_}")
        print(f"Trainable params: {trainable_params:_}")
        print(f"Muon parameters: {num_muon_params:_}")
        print(f"AdamW parameters: {num_other_params:_}")
        print(f"Num optimization steps: {NUM_STEPS}")
        print(f"Device batch size: {BATCH_SIZE}, accumulation: {GRADIENT_ACCUMULATION_STEPS}")
        print(f"Requested batch size (in tokens): {TOKENS_PER_STEP}, effective(in tokens): {BATCH_SIZE * world_size * GRADIENT_ACCUMULATION_STEPS * SEQ_LEN} tokens")

    if RESUME_FROM is None:
        assert not os.path.exists(EXP_OUTPUT_DIR)

    train_rec_loss_history = list()
    train_kldiv_loss_history = list()
    train_grad_norm_history = list()

    ddp_model.train()
    dist.barrier()
    start = time.time()

    torch.cuda.empty_cache()

    for step in range(RESUME_FROM_STEP, NUM_STEPS):
        step_rec_loss = 0
        step_kldiv_loss = 0
        step_elements = 0

        for grad_accumulation_step in range(GRADIENT_ACCUMULATION_STEPS):
            x, src_key_padding_mask, x_unmasked = next(train_loader_iterator)

            x = x.to(DEVICE)
            # mask is [B, 1, L, L]
            src_key_padding_mask = src_key_padding_mask.to(DEVICE)
            x_unmasked = x_unmasked.to(DEVICE)

            x_rec, mu, logvar = ddp_model(x, src_key_padding_mask=src_key_padding_mask)

            with ddp_model.no_sync() if grad_accumulation_step != GRADIENT_ACCUMULATION_STEPS - 1 else nullcontext():
                rec_loss = rec_criterion(x_rec.permute(0, 2, 1), x_unmasked)
                rec_loss[src_key_padding_mask[:, 0, 0]] = 0
                rec_loss = rec_loss.mean()
                kldiv_loss = -0.5 * torch.mean(torch.sum(1 + logvar - mu ** 2 - torch.exp(logvar), dim=-1))
        
                beta = beta_scheduler[step]
                loss = (rec_loss + beta * kldiv_loss) / GRADIENT_ACCUMULATION_STEPS

                loss.backward()

                step_rec_loss += rec_loss.detach() * x.shape[0]
                step_kldiv_loss += kldiv_loss.detach() * x.shape[0]
                step_elements += x.shape[0]

        grad_norm = torch.nn.utils.clip_grad_norm_(other_params, 5.0)

        muon_optimizer.step()
        muon_lr_scheduler.step()
        muon_optimizer.zero_grad()

        other_optimizer.step()
        other_lr_scheduler.step()
        other_optimizer.zero_grad()

        global_rec_loss = reduce_and_get_value(step_rec_loss, step_elements)
        global_kldiv_loss = reduce_and_get_value(step_kldiv_loss, step_elements)
        global_grad_norm = reduce_and_get_value(grad_norm, world_size)

        train_rec_loss_history.append(global_rec_loss)
        train_kldiv_loss_history.append(global_kldiv_loss)
        train_grad_norm_history.append(global_grad_norm)

        if rank == 0 and (step + 1) % TRAIN_LOGGING_FREQ == 0:
            smoothed_rec_loss = round(sum(train_rec_loss_history[-TRAIN_LOGGING_FREQ:]) / TRAIN_LOGGING_FREQ, 4)
            smoothed_kldiv_loss = round(sum(train_kldiv_loss_history[-TRAIN_LOGGING_FREQ:]) / TRAIN_LOGGING_FREQ, 4)
            max_grad_norm = round(max(train_grad_norm_history[-TRAIN_LOGGING_FREQ:]), 4)
            smoothed_grad_norm = round(sum(train_grad_norm_history[-TRAIN_LOGGING_FREQ:]) / TRAIN_LOGGING_FREQ, 4)

            free, total = torch.cuda.mem_get_info(DEVICE)
            mem_used_MB = round((total - free) / 1024 ** 2)

            print(f'{step+1}, elapsed time: {round(time.time() - start, 2)}s, rec loss: {smoothed_rec_loss}, kldiv loss: {smoothed_kldiv_loss}, grad_norm: {smoothed_grad_norm}, max grad norm: {max_grad_norm}, beta: {round(beta, 6)}, mem usage (MB): {mem_used_MB}, Muon lr: {muon_optimizer.param_groups[0]["lr"]}, AdamW lr: {other_optimizer.param_groups[0]["lr"]}')#, end=', ')

        if rank == 0 and (step + 1) % CHECKPOINTING_FREQ == 0:
            if RESUME_FROM is None:
                os.makedirs(EXP_OUTPUT_DIR)
            ddp_model.module.save_pretrained(EXP_OUTPUT_DIR)

            torch.save(muon_optimizer.state_dict(), os.path.join(EXP_OUTPUT_DIR, 'muon_optimizer.pth'))
            torch.save(muon_lr_scheduler.state_dict(), os.path.join(EXP_OUTPUT_DIR, 'muon_lr_scheduler.pth'))

            torch.save(other_optimizer.state_dict(), os.path.join(EXP_OUTPUT_DIR, 'other_optimizer.pth'))
            torch.save(other_lr_scheduler.state_dict(), os.path.join(EXP_OUTPUT_DIR, 'other_lr_scheduler.pth'))

        if (step + 1) % EVAL_FREQ == 0:
            val_rec_loss = 0
            val_kldiv_loss = 0

            ddp_model.eval()
            for eval_step in range(NUM_EVAL_STEPS):
                
                x, src_key_padding_mask, x_unmasked = next(val_loader_iterator)
                x = x.to(DEVICE)
                src_key_padding_mask = src_key_padding_mask.to(DEVICE)
                x_unmasked = x_unmasked.to(DEVICE)

                with torch.no_grad():
                    x_rec, mu, logvar = ddp_model(x, src_key_padding_mask=src_key_padding_mask)

                rec_loss = rec_criterion(x_rec.permute(0, 2, 1), x_unmasked)
                rec_loss[src_key_padding_mask[:, 0, 0]] = 0
                rec_loss = rec_loss.mean()
                kldiv_loss = -0.5 * torch.mean(torch.sum(1 + logvar - mu ** 2 - torch.exp(logvar), dim=-1))

                val_rec_loss += reduce_and_get_value(rec_loss)
                val_kldiv_loss += reduce_and_get_value(kldiv_loss)

            q_mus, q_logvars = get_q_mus_and_logvars(ddp_model.module, val_loader, BATCH_SIZE, DEVICE)
            mu_q = round(reduce_and_get_value(q_mus.mean()), 6)
            logvar_q = round(reduce_and_get_value(q_logvars.mean()), 6)

            val_aup = round(calculate_aup(q_mus, ddp_model.module.config.latent_dim), 2)

            total_time = time.time() - start

            val_rec_loss /= NUM_EVAL_STEPS
            val_kldiv_loss /= NUM_EVAL_STEPS

            ddp_model.train()

            if rank == 0:
                print(f'time: {round(total_time, 2)}s, val rec loss: {round(val_rec_loss, 6)}, val_kldiv_loss: {round(val_kldiv_loss, 6)}, val mu_q: {mu_q}, val logvar_q: {logvar_q}, val AUP: {val_aup}%')

    dist.barrier()     

    if rank == 0:
        ddp_model.module.save_pretrained(EXP_OUTPUT_DIR)

        fig, axes = plt.subplots(ncols=3, figsize=(15, 7))

        axes[0].plot(train_rec_loss_history, label='rec_loss')
        axes[1].plot(train_kldiv_loss_history, label='kl_div_loss')
        axes[2].plot(train_grad_norm_history, label='grad_norm')

        box = np.ones(TRAIN_LOGGING_FREQ)/TRAIN_LOGGING_FREQ
        axes[0].plot(np.convolve(train_rec_loss_history, box, mode='same'), label='rec locc smoothed')
        axes[1].plot(np.convolve(train_kldiv_loss_history, box, mode='same'), label='kldiv loss smoothed')
        axes[2].plot(np.convolve(train_grad_norm_history, box, mode='same'), label='grad norm smoothed')

        axes[0].set_xlabel('step')
        axes[1].set_xlabel('step')
        axes[2].set_xlabel('step')

        axes[0].legend()
        axes[1].legend()
        axes[2].legend()

        fig.savefig(os.path.join(EXP_OUTPUT_DIR, f'graphs.png'))

    dist.barrier()
    dist.destroy_process_group()
    print(f"[rank {rank}] finished")
