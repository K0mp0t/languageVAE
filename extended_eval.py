import os
from argparse import ArgumentParser
import math

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader

from tokenizers import Tokenizer
from datasets import load_dataset

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from tqdm.auto import tqdm

from models import CustomTextVAEConfig, CustomTextVAEModel
from distribution_evaluation import get_p_mus_and_logvars, get_q_mus_and_logvars
from custom_datasets import *


def parse_args():
    parser = ArgumentParser()

    parser.add_argument('model_path', type=str)
    parser.add_argument('tokenizer_path', type=str)

    parser.add_argument('--eval-samples', type=str, default=256)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--seq-len', type=int, default=256)
    parser.add_argument('--dataset-path', type=str, default='./data/fineweb')

    return parser.parse_args()


def plot_hist_with_kde(data, ax, bins=25):
    data_mean = np.mean(data)
    data_std = np.std(data)

    bins = np.linspace(data_mean - 5 * data_std, data_mean + 5 * data_std, bins)
    
    kde = stats.gaussian_kde(data, bw_method='silverman')
    xx = np.linspace(data_mean - 5 * data_std, data_mean + 5 * data_std, len(bins) * 5)
    ax.plot(xx, kde(xx))
    
    ax.hist(data, density=True, bins=bins, alpha=0.3, range=(data_mean - 5 * data_std, data_mean + 5 * data_std))
    
    ax.set_xlim(data_mean - 10 * data_std, data_mean + 10 * data_std)


def gather_mus_logvars_reclosses(loader, model, eval_samples):
    mus = list()
    logvars = list()
    rec_losses = list()

    for x, src_key_padding_mask, x_unmasked in loader:
        x = x.to(DEVICE)
        src_key_padding_mask = src_key_padding_mask.to(DEVICE)
        x_unmasked = x_unmasked.to(DEVICE)

        with torch.no_grad():
            x_rec, mu, logvar = model(x, src_key_padding_mask=src_key_padding_mask)

        mus.append(mu)
        logvars.append(logvar)
        rec_losses.append(F.cross_entropy(x_rec.permute(0, 2, 1), x_unmasked, reduction='none').mean(-1))

        if len(mus) * mus[0].shape[0] >= eval_samples:
            break

    mus = torch.cat(mus).cpu().numpy()
    logvars = torch.cat(logvars).cpu().numpy()
    rec_losses = torch.cat(rec_losses).cpu().numpy()
    
    return mus, logvars, rec_losses


def kl_div_2normal_distributions(mu1, sigma1, mu2, sigma2):
    # IMPORTANT: KL divergence is not symetric so arguments here have the following meaning:
    # D_KL(N(mu1, sigma1^2) || N(mu2, sigma2^2))
    return 0.5 * (((mu1 - mu2) ** 2 / sigma2 ** 2) + ((sigma1 ** 2) / (sigma2 ** 2)) - np.log((sigma1 ** 2) / (sigma2 ** 2)) - 1)


def calculate_kldiv_vs_noise(mus, sigmas, alphas):
    kldivs = list()

    for alpha in alphas:
        mus_ = np.sqrt(alpha) * mus
        sigmas_ = np.sqrt(alpha * np.pow(sigmas, 2) + 1 - alpha)

        mu_ = mus_.mean()
        logvar_ = np.log(np.pow(sigmas_.mean(), 2))

        kldiv = -0.5 * np.mean(np.sum(1 + logvar_ - mu_ ** 2 - np.exp(logvar_), axis=-1))
        kldivs.append(kldiv)

    return kldivs


def calculate_error_vs_noise(loader, model, noise_schedule, eval_samples):
    l2_errors = [list() for i in range(100)]
    ce_errors = [list() for i in range(100)]

    for x, src_key_padding_mask, x_unmasked in loader:
        x = x.to(DEVICE)
        src_key_padding_mask = src_key_padding_mask.to(DEVICE)
        x_unmasked = x_unmasked.to(DEVICE)

        with torch.no_grad():
            emb = model.encode_legacy(x, src_key_padding_mask.logical_not())
            mu, logvar = model.get_distribution_params(emb)
            z = mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)
            x_rec = model.decode(z)

            x_ohe = F.one_hot(x_unmasked, num_classes=model.vocab_size)
            l2_error_0noise = torch.linalg.norm(x_rec - x_ohe)
            ce_error_0noise = F.cross_entropy(x_rec.permute(0, 2, 1), x_unmasked)

            for i, noise_level in enumerate(noise_schedule):
                z_t = torch.sqrt(1 - noise_level) * z + torch.sqrt(noise_level) * torch.randn_like(z)
                x_rec_t = model.decode(z_t)

                l2_errors_ = (torch.linalg.norm(x_ohe - x_rec_t) - l2_error_0noise) / l2_error_0noise
                ce_errors_ = (F.cross_entropy(x_rec_t.permute(0, 2, 1), x_unmasked) - ce_error_0noise) / ce_error_0noise
                l2_errors[i].append(l2_errors_)
                ce_errors[i].append(ce_errors_)

        if len(l2_errors[0]) * x.shape[0] >= eval_samples:
            break

    l2_errors = torch.as_tensor(l2_errors).mean(dim=-1).cpu().numpy()
    ce_errors = torch.as_tensor(ce_errors).mean(dim=-1).cpu().numpy()

    return l2_errors, ce_errors


def calculate_aup_and_au_indices(mus, active_threshold=1e-2):
    mu_stds = np.std(mus.reshape(-1, mus.shape[-1]), axis=0, correction=1)
    aup = sum(mu_stds > active_threshold).item() / mus.shape[-1] * 100

    au_indices = np.where(mu_stds > active_threshold)[0]

    return aup, au_indices, mu_stds


if __name__ == '__main__':
    args = parse_args()

    MODEL_PATH = args.model_path
    TOKENIZER_PATH = args.tokenizer_path

    EVAL_SAMPLES = args.eval_samples
    BATCH_SIZE = args.batch_size
    DATASET_PATH = args.dataset_path
    SEQ_LEN = args.seq_len

    OUTPUT_PATH = os.path.join(MODEL_PATH, 'eval_output')
    os.makedirs(OUTPUT_PATH, exist_ok=True)

    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    tokenizer = Tokenizer.from_file(TOKENIZER_PATH)
    tokenizer.add_special_tokens(["<|mask|>"])

    if any(fn.endswith('.parquet') for fn in os.listdir(DATASET_PATH)):
        train_loader = DDPStreamingParquetDataLoader(tokenizer, DATASET_PATH, SEQ_LEN, batch_size=BATCH_SIZE, split='train')
        val_loader = DDPStreamingParquetDataLoader(tokenizer, DATASET_PATH, SEQ_LEN, batch_size=BATCH_SIZE, split='validation')
    else:    
        train_loader = DDPStreamingParquetDataLoader(tokenizer, os.path.join(DATASET_PATH, 'validation'), SEQ_LEN, batch_size=BATCH_SIZE, split='all')
        val_loader = DDPStreamingParquetDataLoader(tokenizer, os.path.join(DATASET_PATH, 'validation'), SEQ_LEN, batch_size=BATCH_SIZE, split='all')

    model = CustomTextVAEModel.from_pretrained(MODEL_PATH)
    model.to(DEVICE)
    model.eval()

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(
        p.numel() for name, p in model.named_parameters()
        if p.requires_grad
    )
    print(f"Total params:   {total_params:,}")
    print(f"Trainable params: {trainable_params:,}")

    print(f'Calculating distribution parameters and reconstruction losses for {EVAL_SAMPLES} samples')

    train_mus, train_logvars, train_rec_losses = gather_mus_logvars_reclosses(train_loader, model, EVAL_SAMPLES)
    val_mus, val_logvars, val_rec_losses = gather_mus_logvars_reclosses(val_loader, model, EVAL_SAMPLES)
    train_sigmas = np.exp(0.5 * train_logvars)
    val_sigmas = np.exp(0.5 * val_logvars)
    
    train_average_mu = train_mus.mean()
    train_average_sigma = train_sigmas.mean()
    val_average_mu = val_mus.mean()
    val_average_sigma = val_sigmas.mean()

    fig, axes = plt.subplots(ncols=3, nrows=2, figsize=(20, 12))

    plot_hist_with_kde(train_mus.flatten(), axes[0, 0])
    plot_hist_with_kde(train_logvars.flatten(), axes[0, 1])
    plot_hist_with_kde(train_rec_losses.flatten(), axes[0, 2])

    plot_hist_with_kde(val_mus.flatten(), axes[1, 0])
    plot_hist_with_kde(val_logvars.flatten(), axes[1, 1])
    plot_hist_with_kde(val_rec_losses.flatten(), axes[1, 2])

    axes[0, 0].set_title('train mu distribution')
    axes[0, 1].set_title('train logvar distribution')
    axes[0, 2].set_title('train rec loss distribution')

    axes[1, 0].set_title('val mu distribution')
    axes[1, 1].set_title('val logvar distribution')
    axes[1, 2].set_title('val rec loss distribution')

    fig_output_path = os.path.join(OUTPUT_PATH, 'distributions.png')
    print(f'Saving distribution parameters and reconstruction losses to {fig_output_path}')
    fig.savefig(fig_output_path)


    train_kldiv = np.mean(np.array([kl_div_2normal_distributions(mu, sigma, train_average_mu, train_average_sigma) for mu, sigma in zip(train_mus, train_sigmas)]))
    print('train D_{KL}(q(z_i|x_i)||q(z)) (how each sample is different from all samples on average):', train_kldiv)

    val_kldiv = np.mean(np.array([kl_div_2normal_distributions(mu, sigma, val_average_mu, val_average_sigma) for mu, sigma in zip(val_mus, val_sigmas)]))
    print('val D_{KL}(q(z_i|x_i)||q(z)) (how each sample is different from all samples on average):', val_kldiv)


    alphas = np.linspace(1, 0, 20)
    train_kldivs = calculate_kldiv_vs_noise(train_mus, train_sigmas, alphas)
    val_kldivs = calculate_kldiv_vs_noise(val_mus, val_sigmas, alphas)

    fig, axes = plt.subplots(ncols=2, figsize=(15, 5))

    axes[0].plot(alphas[::-1], train_kldivs)
    axes[1].plot(alphas[::-1], val_kldivs)

    axes[0].set_title('train K_DL(z_t||N(0,1)) vs noise level')
    axes[1].set_title('val D_KL(z_t||N(0,1)) vs noise level')

    fig_output_path = os.path.join(OUTPUT_PATH, 'kldiv_vs_noise_level.png')
    print(f'Saving K_DL(z_t||N(0,1)) vs noise level plots to {fig_output_path}')   
    fig.savefig(fig_output_path)


    train_aup, train_au_indices, train_mu_stds = calculate_aup_and_au_indices(train_mus)
    val_aup, val_au_indices, val_mu_stds = calculate_aup_and_au_indices(val_mus)
    
    print('train AU%', train_aup)
    print('val AU%', val_aup)

    fig, axes = plt.subplots(ncols=2, figsize=(15, 6))

    latent_size = train_mus.shape[-1]
    for e in range(int(math.sqrt(latent_size)), 0, -1):
        if latent_size % e == 0:
            optimal_factor = e
            break

    axes[0].matshow(train_mu_stds.reshape(optimal_factor, latent_size // optimal_factor))
    axes[1].matshow(val_mu_stds.reshape(optimal_factor, latent_size // optimal_factor))

    axes[0].set_title('train active units')
    axes[1].set_title('val active units')

    fig_output_path = os.path.join(OUTPUT_PATH, 'active_units.png')
    print(f'Saving active unit plots to {fig_output_path}')   
    fig.savefig(fig_output_path)


    noise_schedule = torch.linspace(0, 0.5, 100)
    train_l2_errors, train_ce_errors = calculate_error_vs_noise(train_loader, model, noise_schedule, EVAL_SAMPLES)
    val_l2_errors, val_ce_errors = calculate_error_vs_noise(val_loader, model, noise_schedule, EVAL_SAMPLES)

    fig, axes = plt.subplots(ncols=2, nrows=2, figsize=(15, 6))

    axes[0, 0].plot(noise_schedule, train_l2_errors)
    axes[0, 1].plot(noise_schedule, val_l2_errors)
    axes[0, 0].set_xlabel('noise level')
    axes[0, 1].set_xlabel('noise level')
    axes[0, 0].set_ylabel('L2 error level')
    axes[0, 1].set_ylabel('L2 error level')

    axes[1, 0].plot(noise_schedule, train_ce_errors)
    axes[1, 1].plot(noise_schedule, val_ce_errors)
    axes[1, 0].set_xlabel('noise level')
    axes[1, 1].set_xlabel('noise level')
    axes[1, 0].set_ylabel('CE level')
    axes[1, 1].set_ylabel('CE level')

    fig_output_path = os.path.join(OUTPUT_PATH, 'reconstruction_error_vs_noise_level.png')
    print(f'Saving reconstruction error vs noise level plots to {fig_output_path}')   
    fig.savefig(fig_output_path)
