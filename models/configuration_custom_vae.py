from transformers import PretrainedConfig
from typing import Type
from torch import nn
from .mergers import *


class CustomTextVAEConfig(PretrainedConfig):
    """Configuration for :class:`CustomTextVAEModel`.

    Parameterizes the **Custom Text VAE** of Cola DLM, which provides *both*
    the inference encoder ``q_phi(z_0 | x)`` and the conditional
    decoder ``p_theta(x | z_0)`` of the joint factorization

        p(x, z_0) = p_theta(x | z_0) * p_psi(z_0)

    (Eq. 2.1.1 of the paper *Continuous Latent Diffusion Language
    Model*, arXiv:2605.06548). Stage-1 pretraining fits this module
    with the objective ``L_VAE`` of Eq. 2.2.1; in Stage 2 the same
    encoder/decoder are jointly trained with the DiT prior.

    Key knobs (in paper notation):

    * ``vocab_size``: tokenizer vocabulary (OLMo 2 BPE in the released
      checkpoint).
    * ``encoder_num_blocks`` / ``decoder_num_blocks`` / ``dim`` /
      ``ffn_dim`` / ``num_heads``: shape of the transformer trunk
      backing ``q_phi`` and ``p_theta``.
    * ``latent_dim``: dimension ``d`` of the continuous latent
      ``z_0 ∈ R^d`` (Eq. 2.1.1). Must agree with
      :class:`ColaDiTConfig.txt_in_channels` / ``txt_out_channels``.
    * ``patch_size``: 1-D patchification factor between the token axis
      and the latent axis (``n_i = L_i / patch_size``).
    * ``block_causal`` / ``block_size``: per-sample factor that defines
      how the latent sequence is partitioned into ``B`` blocks
      ``z_0 = (z_0^(1), ..., z_0^(B))`` (Eq. 2.1.4). The same
      ``block_size`` is used by the DiT prior to enforce the visible
      set ``V_b`` (Eq. 2.2.3); both VAE encoder and decoder respect
      this factorization to prevent information leakage and to keep
      streaming generation well-defined.
    * ``rope_theta`` / ``rope_full_precision``: RoPE positional
      encoding configuration.
    * ``use_variation``: whether to materialize ``q_phi`` as a Gaussian
      posterior (mean + log-variance) versus a deterministic encoder.
    * ``scaling_factor`` / ``shifting_factor``: per-channel
      normalization applied to VAE latents at the boundary with the
      DiT prior, ``z_0 ← (z_0 - shifting_factor) * scaling_factor``.
      They control the geometry of the latent space over which
      ``p_psi`` is learned.
    """

    model_type = "custom_text_vae"
    allowed_mergers = {
        'mlp': (MLPPatchMerger, MLPPatchUnMerger),
        'linear': (LinearPatchMerger, LinearPatchUnMerger),
        'sum': (SumPatchMerger, SumPatchUnMerger)
    }

    def __init__(
        self,
        vocab_size: int = 100279,
        encoder_num_blocks: int = 5,
        decoder_num_blocks: int = 5,
        dim: int = 1536,
        latent_dim: int = None,
        patch_size: int = 1,
        compression_config: list = None,
        num_heads: int = 8,
        layer_norm_type: Type = "layer_norm",
        layer_norm_eps: float = 1e-6,
        rope_theta: int = 500000,
        dropout: float = 0.0,
        act: Type = "silu",
        init_fn: str = "normal",
        init_std: float = 0.02,
        init_cutoff_factor: float = 3,
        use_variation: bool = True,
        use_emb: bool = True,
        scaling_factor: float = 1.0,
        shifting_factor: float = 0.0,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.encoder_num_blocks = encoder_num_blocks
        self.decoder_num_blocks = decoder_num_blocks
        self.dim = dim
        self.latent_dim = latent_dim
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.layer_norm_type = layer_norm_type
        self.layer_norm_eps = layer_norm_eps
        self.rope_theta = rope_theta
        self.dropout = dropout
        self.act = act
        self.init_fn = init_fn
        self.init_std = init_std
        self.init_cutoff_factor = init_cutoff_factor
        self.use_variation = use_variation
        self.use_emb = use_emb
        self.scaling_factor = scaling_factor
        self.shifting_factor = shifting_factor

        assert encoder_num_blocks == decoder_num_blocks

        if compression_config is None:
            self.compression_config = list()
            for i in range(self.encoder_num_blocks):
                if i % 2 == 0:
                    sequence_merge_ratio = 1
                    input_size = self.dim // (2 ** (i // 2))
                    output_size = self.dim // (2 ** (i // 2 + 1))
                else:
                    sequence_merge_ratio = 2
                    input_size = self.dim // (2 ** (i // 2))
                    output_size = self.dim // (2 ** (i // 2))

                self.compression_config.append([sequence_merge_ratio, input_size, output_size, 'mlp', 'mlp'])
        else:
            assert all(e[3] in self.allowed_mergers and e[4] in self.allowed_mergers for e in compression_config)
            self.compression_config = compression_config

        if self.latent_dim is None:
            self.latent_dim = self.compression_config[-1][2]
        else:
            self.compression_config[-1][-1] = self.latent_dim

        super().__init__(**kwargs)
