import torch
from torch import nn
from torch.nn import functional as F
from tqdm import tqdm

import time


def get_q_mus_and_logvars(model, dataloader, samples_to_take, device):
    mus = list()
    logvars = list()

    model.to(device)
    model.eval()
    if isinstance(model, nn.Module):
        model.disable_grad()
    else:
        model.module.disable_grad()

    for x, attention_mask, x_unmasked in dataloader:
        x_unmasked = x_unmasked.to(device)
        attention_mask = attention_mask.to(device)

        with torch.no_grad():
            hidden = model.encode_legacy(x_unmasked, attention_mask)

            mu, logvar = model.get_distribution_params(hidden) # [B, L, E]

        mus.append(mu.flatten())
        logvars.append(logvar.flatten())

        if len(mus) * mus[0].shape[0] >= samples_to_take:
            break

    mus = torch.cat(mus)
    logvars = torch.cat(logvars)

    model.train()
    if isinstance(model, nn.Module):
        model.enable_grad()
    else:
        model.module.enable_grad()

    return mus, logvars


def log_posterior(z, x, model):
    """
    z: [B, L, latent_dim]
    x: [B, L, E]
    """

    logits = model.decode(z)

    log_px_z = -F.cross_entropy(
        logits.permute(0, 2, 1),
        x,
        reduction="sum"
    )
    log_pz = -0.5 * (z ** 2).sum()

    return log_px_z + log_pz


def langevein_chain_generator(model, x, latent_shape, max_samples, device, eps=0.001):
    z = torch.randn(x.shape[0], *latent_shape, device=device)

    for _ in range(max_samples):
        z.requires_grad_(True)
        logp = log_posterior(z, x, model)

        grad = torch.autograd.grad(logp, z)[0]

        with torch.no_grad():
            z = (
                z
                + 0.5 * eps * grad
                + torch.sqrt(torch.tensor(eps)) * torch.randn_like(z)
            )

        yield z

def get_p_mus_and_logvars(model, dataloader, latent_shape, samples_to_take, device, max_samples=1000, num_chains=3, spread_check_interval=25):
    model.to(device)
    model.eval()
    model.disable_grad()

    mus = list()
    logvars = list()

    for x, attention_mask, x_unmasked in dataloader:
        start = time.time()

        x_unmasked = x_unmasked.to(device)
        chain_generators = [
            langevein_chain_generator(model, x_unmasked, latent_shape, max_samples, device) for _ in range(num_chains)
        ]

        good_spread_count = 0
        zs = list()

        for i in range(max_samples):
            chain_samples = torch.stack(list(next(g) for g in chain_generators), dim=1) # [B, chains, L, E]
            # chain_samples = chain_samples.mean(dim=2) # [B, chains, E]
            zs.append(chain_samples) # [B, chains, L, E]

            if i % spread_check_interval == 0:
                spread = torch.stack(zs[-spread_check_interval:]).std(dim=2).mean().item() # [S, B, chains, E] -> [S, B, E] -> scalar

                if spread < 0.05:
                    good_spread_count += 1
                else:
                    good_spread_count = 0
                    zs = list()
                
                if good_spread_count > 1:
                    break

        print(f'sampling took {i} steps and {round(time.time() - start, 2)} seconds')

        mus.append(torch.stack(zs[-spread_check_interval:]).mean(dim=(0, 2, 3))) # [B,]
        logvars.append(torch.stack(zs[-spread_check_interval:]).std(dim=(0, 2, 3))) # [B,]

        if len(mus) * mus[0].shape[0] >= samples_to_take:
            break

    model.train()
    model.enable_grad()

    mus = torch.cat(mus).cpu()
    logvars = torch.cat(logvars).cpu()

    return mus, logvars


def calculate_aup(mus, latent_size, active_threshold=1e-2):
    mu_stds = torch.std(mus.reshape(-1, latent_size), dim=0, correction=1)
    aup = torch.sum(mu_stds > active_threshold).item() / latent_size * 100

    return aup
