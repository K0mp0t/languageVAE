# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Cola Text VAE — HuggingFace Transformers compatible, NA (no-padding) inference.

This module implements the **Text VAE** of Cola DLM, which provides
*both* the inference encoder ``q_phi(z_0 | x)`` and the conditional
decoder ``p_theta(x | z_0)`` of the joint factorization

    p(x, z_0) = p_theta(x | z_0) * p_psi(z_0)

(Eq. 2.1.1 of the paper *Continuous Latent Diffusion Language Model*,
arXiv:2605.06548). The encoder is *not* part of the generative model —
it is used at training time for variational inference and at inference
time to encode the prefix into clean latent conditions
``z^pre ~ q_phi(z^pre | x^pre)`` (Eq. 2.2.4).

In the Stage-1 pretraining pipeline of the paper, this module is
trained with (Eq. 2.2.1)::

    L_VAE = - E_{q_phi(z_0|x)}[log p_theta(x | z_0)]
            + beta * KL(q_phi(z_0 | x) || p_base(z_0))
            + lambda_mask * L_mask,

where ``L_mask`` is a BERT-style masking loss that prevents the encoder
from collapsing semantically while the decoder merely memorizes surface
text. In Stage 2, the same encoder/decoder are jointly trained with the
DiT prior under a reference-encoder KL regularizer that suppresses
latent drift; see ``docs/architecture.md`` §9 for the explicit
objectives. Training code is not included in this open-source release.

The public API is a list-in / list-out variant:

* :meth:`ColaTextVAEModel.encode` accepts ``List[LongTensor(L_i,)]`` —
  one ``input_ids`` tensor per sample, each of length already divisible
  by ``patch_size * block_size`` (the per-sample pad is kept because
  ``nn.Conv1d`` cannot handle variable-length inputs; everything else
  happens in flattened NA form).
* :meth:`ColaTextVAEModel.decode` works in flattened NA form with
  per-sample ``txt_shape`` / ``txt_q_shape`` describing K and Q
  lengths, and realizes ``hat x^res ~ p_theta(x^res | z^pre, hat z_0^(1:B))``
  (Eq. 2.2.6) one block at a time.
"""

import math
from collections.abc import MutableMapping
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn
from transformers import AutoConfig, AutoModel, PreTrainedModel

from .configuration_custom_vae import CustomTextVAEConfig
from .mergers import *

# ---------------------------------------------------------------------------
# DiagonalGaussianDistribution
# ---------------------------------------------------------------------------


class DiagonalGaussianDistribution:
    def __init__(self, parameters: torch.Tensor, deterministic: bool = False):
        assert parameters.ndim in (2, 3)
        self.parameters = parameters
        self.mean, self.logvar = torch.chunk(parameters, 2, dim=-1)
        self.logvar = torch.clamp(self.logvar, -30.0, 20.0)
        self.deterministic = deterministic
        self.std = torch.exp(0.5 * self.logvar)
        self.var = torch.exp(self.logvar)
        if self.deterministic:
            self.var = self.std = torch.zeros_like(
                self.mean, device=self.parameters.device, dtype=self.parameters.dtype
            )

    def sample(self, max_std: float = 1, generator: Optional[torch.Generator] = None) -> torch.Tensor:
        sample = torch.randn(
            self.mean.shape,
            generator=generator,
            device=self.parameters.device,
            dtype=self.parameters.dtype,
        )
        return self.mean + self.std.clamp_max(max_std) * sample

    def mode(self) -> torch.Tensor:
        return self.mean


# ---------------------------------------------------------------------------
# Encoder / Decoder output dataclasses
# ---------------------------------------------------------------------------


@dataclass
class TextVAEEncoderOutput:
    # ``latents_list[i]`` has shape ``(n_i, latent_dim)``.
    latents_list: list[torch.Tensor]
    # ``latent_dists[i]`` is the posterior over sample ``i`` (None when
    # ``use_variation=False``).
    latent_dists: Optional[list[DiagonalGaussianDistribution]] = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class BufferCache(dict, MutableMapping[str, torch.Tensor]):
    pass


def _build_na_positions(txt_shape: torch.LongTensor) -> torch.Tensor:
    """``txt_shape: (B, 1)`` → ``positions: (1, L_total)`` where each
    sample's positions restart from 0."""
    parts = [torch.arange(int(l), device=txt_shape.device) for l in txt_shape.flatten()]
    return torch.cat(parts).unsqueeze(0)


def _build_na_q_positions(txt_shape: torch.LongTensor, txt_q_shape: torch.LongTensor) -> torch.Tensor:
    """Per-sample Q positions aligned to the TAIL of K within each sample."""
    parts = []
    for k_len, q_len in zip(txt_shape.flatten().tolist(), txt_q_shape.flatten().tolist()):
        parts.append(torch.arange(k_len - q_len, k_len, device=txt_shape.device))
    return torch.cat(parts).unsqueeze(0)


def init_normal(module, std: float, init_cutoff_factor: Optional[float] = None):
    if init_cutoff_factor is not None:
        cutoff_value = init_cutoff_factor * std
        nn.init.trunc_normal_(module.weight, mean=0.0, std=std, a=-cutoff_value, b=cutoff_value)
    else:
        nn.init.normal_(module.weight, mean=0.0, std=std)
    if isinstance(module, nn.Linear) and module.bias is not None:
        nn.init.zeros_(module.bias)


# ---------------------------------------------------------------------------
# Rotary Embedding (not NA-capable)
# ---------------------------------------------------------------------------


class RotaryPositionEmbedding(nn.Module):
    def __init__(self, hidden_size, num_heads, rope_theta=10000):
        super().__init__()
        assert hidden_size % num_heads == 0
        
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        self.rope_theta = rope_theta

        self.thetas = torch.pow(rope_theta, -2 * torch.arange(self.head_dim // 2) / self.head_dim)

        # [1024, H/2]
        self.sines = torch.sin(self.thetas.unsqueeze(0) * torch.arange(1024).unsqueeze(1))
        self.cosines = torch.cos(self.thetas.unsqueeze(0) * torch.arange(1024).unsqueeze(1))


    def forward(self, query, key):
        _, _, q_seq_len, _ = query.shape
        _, _, k_seq_len, _ = key.shape
        device = query.device

        if k_seq_len > self.sines.shape[0] or self.sines.is_meta:
            self.thetas = torch.pow(self.rope_theta, -2 * torch.arange(self.head_dim // 2) / self.head_dim)

            self.sines = torch.sin(self.thetas.unsqueeze(0) * torch.arange(k_seq_len).unsqueeze(1))
            self.cosines = torch.cos(self.thetas.unsqueeze(0) * torch.arange(k_seq_len).unsqueeze(1))

        if self.sines.device != device:
            self.sines = self.sines.to(device)
            self.cosines = self.cosines.to(device)

        query_real, query_imag = query.float().reshape(query.shape[:-1] + (-1, 2)).unbind(-1)
        key_real, key_imag = key.float().reshape(key.shape[:-1] + (-1, 2)).unbind(-1)

        query_real_rotated = torch.zeros_like(query, device=device)
        query_imag_rotated = torch.zeros_like(query, device=device)
        key_real_rotated = torch.zeros_like(key, device=device)
        key_imag_rotated = torch.zeros_like(key, device=device)

        query_real_rotated[..., 0::2] = query_real * self.cosines[:q_seq_len]
        query_real_rotated[..., 1::2] = query_real * self.sines[:q_seq_len]

        query_imag_rotated[..., 0::2] = -query_imag * self.sines[:q_seq_len]
        query_imag_rotated[..., 1::2] = query_imag * self.cosines[:q_seq_len]

        key_real_rotated[..., 0::2] = key_real * self.cosines[:k_seq_len]
        key_real_rotated[..., 1::2] = key_real * self.sines[:k_seq_len]

        key_imag_rotated[..., 0::2] = -key_imag * self.sines[:k_seq_len]
        key_imag_rotated[..., 1::2] = key_imag * self.cosines[:k_seq_len]

        query_out = query_real_rotated + query_imag_rotated
        key_out = key_real_rotated + key_imag_rotated

        return query_out, key_out
        

# ---------------------------------------------------------------------------
# Attention
# ---------------------------------------------------------------------------


class MultiHeadAttentionWithRoPE(nn.Module):
    def __init__(self,
                 hidden_size,
                 num_heads,
                 bias=True,
                 rope_theta=10000.0):
        super().__init__()
        if hidden_size % num_heads != 0:
            raise ValueError(
                '`hidden_size`({}) should be divisible by `num_heads`({})'.format(hidden_size, num_heads))
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.bias = bias
        self.linear_q = nn.Linear(hidden_size, hidden_size, bias)
        self.linear_k = nn.Linear(hidden_size, hidden_size, bias)
        self.linear_v = nn.Linear(hidden_size, hidden_size, bias)
        self.linear_o = nn.Linear(hidden_size, hidden_size, bias)

        self.rope = RotaryPositionEmbedding(hidden_size, num_heads, rope_theta)

        self.batch_first = True
        self.head_dim = hidden_size // num_heads

    def forward(self, q, k, v, attention_mask=None):
        # B, L, E
        q, k, v = self.linear_q(q), self.linear_k(k), self.linear_v(v)

        q = self._reshape_to_batches(q)
        k = self._reshape_to_batches(k)
        v = self._reshape_to_batches(v)

        # B, H, L, e
        q, k = self.rope(q, k)

        y = F.scaled_dot_product_attention(q, k, v, attention_mask)

        y = self._reshape_from_batches(y)
        # B, L, E
        y = self.linear_o(y)
        return y

    def _reshape_to_batches(self, x):
        batch_size, seq_len, hidden_size = x.size()
        sub_dim = hidden_size // self.num_heads
        return x.reshape(batch_size, seq_len, self.num_heads, sub_dim).permute(0, 2, 1, 3)

    def _reshape_from_batches(self, x):
        batch_size, heads, seq_len, sub_dim = x.size()
        hidden_size = sub_dim * heads
        return x.permute(0, 2, 1, 3).reshape(batch_size, seq_len, hidden_size)  


# ---------------------------------------------------------------------------
# TextVAEBlock — transformer block for encoder/decoder (NA form)
# ---------------------------------------------------------------------------


class CustomTransformerEncoderLayer(nn.Module):
    def __init__(self, hidden_size, num_heads, dim_feedforward=2048, output_size=None, dropout=0.1, activation=nn.SiLU(), layer_norm_type=nn.LayerNorm, layer_norm_eps=1e-6, rope_theta=10000):
        super().__init__()
        self.self_attn = MultiHeadAttentionWithRoPE(hidden_size, num_heads, rope_theta=rope_theta)

        self.linear1 = nn.Linear(hidden_size, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, hidden_size if output_size is None else output_size)

        self.linear3 = None
        if output_size is not None and output_size != hidden_size:
            self.linear3 = nn.Linear(hidden_size, output_size)

        self.norm1 = layer_norm_type(hidden_size, eps=layer_norm_eps)
        self.norm2 = layer_norm_type(hidden_size if output_size is None else output_size, eps=layer_norm_eps)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = activation


    def forward(self, src, attention_mask=None):
        src2 = self.self_attn(src, src, src, attention_mask=attention_mask)
        src2 = self.norm1(src2)
        src = src + self.dropout1(src2)

        src2 = self.linear2(self.activation(self.linear1(src)))
        src2 = self.norm2(src2)
        if self.linear3 is not None:
            src = self.linear3(src)
        src = src + self.dropout2(src2)
        return src  


# ---------------------------------------------------------------------------
# Main Model: ColaTextVAEModel
# ---------------------------------------------------------------------------


class CustomTextVAEModel(PreTrainedModel):
    config_class = CustomTextVAEConfig
    base_model_prefix = "text_vae"

    def __init__(self, config: CustomTextVAEConfig):
        super().__init__(config)

        self.latent_dim = config.latent_dim
        self.compression_config = config.compression_config
        self.num_heads = config.num_heads
        self.vocab_size = config.vocab_size
        self.dropout = config.dropout
        self.layer_norm_eps = config.layer_norm_eps
        self.rope_theta = config.rope_theta
        self.patch_size = config.patch_size
        self.use_variation = config.use_variation
        self.scaling_factor = config.scaling_factor
        self.shifting_factor = config.shifting_factor
        
        if config.act == 'silu':
            self.act = nn.SiLU()
        elif config.act == 'relu':
            self.act = nn.ReLU()
        elif config.act == 'gelu':
            self.act = nn.GeLU()
        else:
            raise ValueError(f'unsupported activation {config.act}')

        if config.layer_norm_type == 'layer_norm':
            self.layer_norm_type = nn.LayerNorm
        elif config.layer_norm_type == 'batch_norm':
            self.layer_norm_type = nn.BatchNorm1d
        else:
            raise ValueError(f'unsupported norm type {config.layer_norm_type}')

        self.input_embedding = nn.Embedding(self.vocab_size, self.compression_config[0].input_size)

        self.mu_proj = nn.Linear(self.latent_dim, self.latent_dim)
        self.sigma_proj = nn.Linear(self.latent_dim, self.latent_dim)

        encoder_blocks = list()
        decoder_blocks = list()

        encoder_mergers = list()
        decoder_unmergers = list()

        for item in self.compression_config:
            encoder_blocks.append(
                CustomTransformerEncoderLayer(item.input_size, num_heads=self.num_heads, dim_feedforward=item.input_size*4, output_size=item.output_size, 
                dropout=self.dropout, activation=self.act, layer_norm_type=self.layer_norm_type, layer_norm_eps=self.layer_norm_eps, rope_theta=self.rope_theta)
            )
            encoder_mergers.append(CustomTextVAEConfig.allowed_mergers[item.merger_name][0](item.merge_ratio, item.output_size, item.merge_window_width))

            decoder_blocks.append(
                CustomTransformerEncoderLayer(item.output_size, num_heads=self.num_heads, dim_feedforward=item.output_size*4, output_size=item.input_size, 
                dropout=self.dropout, activation=self.act, layer_norm_type=self.layer_norm_type, layer_norm_eps=self.layer_norm_eps, rope_theta=self.rope_theta)
            )

            decoder_unmergers.append(
                CustomTextVAEConfig.allowed_mergers[item.unmerger_name][1](item.merge_ratio, item.output_size, item.unmerge_window_width)
            )

        self.encoder_blocks = nn.ModuleList(encoder_blocks)
        self.decoder_blocks = nn.ModuleList(decoder_blocks[::-1])
        self.encoder_mergers = nn.ModuleList(encoder_mergers)
        self.decoder_unmergers = nn.ModuleList(decoder_unmergers[::-1])

        self.sequence_compression_ratio = math.prod(e.merge_ratio for e in self.compression_config)

        self.output_proj = nn.Linear(self.compression_config[0].input_size, self.vocab_size)

        self.post_init()

    
    def _init_weights(self, module):
        std = self.config.init_std
        cutoff = self.config.init_cutoff_factor
        if isinstance(module, (nn.Linear, nn.Embedding)):
            init_normal(module, std, cutoff)
        if isinstance(module, nn.LayerNorm):
            if module.weight is not None:
                nn.init.ones_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)


    def encode_legacy(self, x, attention_mask=None):
        emb = self.input_embedding(x)

        for block, merger in zip(self.encoder_blocks, self.encoder_mergers):
            emb = block(emb, attention_mask)
            emb, attention_mask = merger(emb, attention_mask)

        return emb


    def encode(self, x, attention_mask=None):
        if attention_mask is not None:
            if isinstance(attention_mask, torch.Tensor):
                mutable_attention_mask = attention_mask.clone()
            else:
                mutable_attention_mask = attention_mask.copy()
        else:
            mutable_attention_mask = None
        
        if isinstance(x, list):
            output = list()

            for i, e in enumerate(x):
                attention_mask_ = attention_mask[i].unsqueeze(0) if attention_mask is not None else None

                emb = self.input_embedding(e.unsqueeze(0))

                for j, (block, merger) in enumerate(zip(self.encoder_blocks, self.encoder_mergers)):
                    emb = block(emb, attention_mask_)
                    emb, attention_mask_ = merger(emb, attention_mask_)

                mean, logvar = self.get_distribution_params(emb)
                logvar = torch.clamp(logvar, -10, 10)
                output.append(torch.cat((mean, logvar), dim=-1).squeeze(0))

                if mutable_attention_mask is not None:
                    mutable_attention_mask[i] = attention_mask_

        else:
            emb = self.input_embedding(x)

            for block, merger in zip(self.encoder_blocks, self.encoder_mergers):
                emb = block(emb, mutable_attention_mask)
                emb, mutable_attention_mask = merger(emb, mutable_attention_mask)

            mean, logvar = self.get_distribution_params(emb)
            logvar = torch.clamp(logvar, -10, 10)
            output = torch.cat((mean, logvar), dim=-1)

        latent_dists: Optional[list[DiagonalGaussianDistribution]] = None
        if self.use_variation:
            latent_dists = [DiagonalGaussianDistribution(p) for p in output]
            latents_mode = [d.mode() for d in latent_dists]  # mean only
        else:
            latents_mode = mean

        return TextVAEEncoderOutput(latents_list=latents_mode, latent_dists=latent_dists)

    
    def get_distribution_params(self, hidden):
        mu = self.mu_proj(hidden)
        logvar = self.sigma_proj(hidden)

        # if torch.abs(logvar).max().item() > 10:
        #     print(f'[LOGVAR WARNING] {round(torch.abs(logvar).max().item(), 4)}, {torch.sum(torch.abs(logvar) > 10)}')
        logvar = torch.clamp(logvar, -10, 10)

        return mu, logvar

    
    def decode(self, z):
        if isinstance(z, list):
            out = list()

            for i, e in enumerate(z):
                emb = e.unsqueeze(0)

                for block, unmerger in zip(self.decoder_blocks, self.decoder_unmergers):
                    emb = block(emb)
                    emb = unmerger(emb)

                out.append(self.output_proj(emb).squeeze(0))

        else:
            emb = z

            for block, unmerger in zip(self.decoder_blocks, self.decoder_unmergers):
                emb = block(emb)
                emb = unmerger(emb)
                
            out = self.output_proj(emb)

        return out


    def forward_with_a_list(self, x_list, attention_masks_list=None, src_key_padding_masks_list=None, variance=True):
        resulting_attention_masks_list = list()

        if attention_masks_list is None:
            attention_masks_list = [None] * len(x_list)

        if src_key_padding_masks_list is None:
            src_key_padding_masks_list = [None] * len(x_list)

        for attention_mask, src_key_padding_mask in zip(attention_masks_list, src_key_padding_masks_list):
            if attention_mask is not None:
                if src_key_padding_mask is not None:
                    resulting_attention_mask = torch.logical_and(attention_mask, torch.logical_not(src_key_padding_mask))
                else:
                    resulting_attention_mask = attention_mask
            elif src_key_padding_mask is not None:
                resulting_attention_mask = torch.logical_not(src_key_padding_mask)
            else:
                resulting_attention_mask = None

            resulting_attention_masks_list.append(resulting_attention_mask)

        encoder_output = self.encode(x_list, resulting_attention_masks_list)

        # distribution_params = [self.get_distribution_params(e.unsqueeze(0)) for e in emb]
        if variance:
            zs = [e.sample() for e in encoder_output.latent_dists]
        else:
            zs = encoder_output.latents_list

        out = self.decode(zs)

        return out


    def forward(self, x, attention_mask=None, src_key_padding_mask=None):
        if attention_mask is not None:
            if src_key_padding_mask is not None:
                resulting_attention_mask = torch.logical_and(attention_mask, torch.logical_not(src_key_padding_mask))
            else:
                resulting_attention_mask = attention_mask
        elif src_key_padding_mask is not None:
            resulting_attention_mask = torch.logical_not(src_key_padding_mask)
        else:
            resulting_attention_mask = None
        
        emb = self.encode_legacy(x, resulting_attention_mask)

        mu, logvar = self.get_distribution_params(emb)

        z = mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)

        out = self.decode(z)

        return out, mu, logvar

    @property
    def device(self):
        return next(self.parameters()).device


    def disable_grad(self):
        for p in self.parameters():
            p.requires_grad_(False)


    def enable_grad(self):
        for p in self.parameters():
            p.requires_grad_(True)


AutoConfig.register("custom_text_vae", CustomTextVAEConfig)
AutoModel.register(CustomTextVAEConfig, CustomTextVAEModel)
