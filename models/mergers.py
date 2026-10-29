import math
import torch
from torch import nn
from torch.nn import functional as F


class LinearPatchMerger(nn.Module):
    def __init__(self, merge_ratio, hidden_size, merge_window_width):
        super().__init__()

        self.hidden_size = hidden_size
        self.merge_ratio = merge_ratio
        self.merge_window_width = merge_window_width

        if self.merge_ratio != 1:
            self.linear = nn.Linear(hidden_size * merge_window_width, hidden_size * merge_window_width // merge_ratio)

    def forward(self, x, attention_mask=None):
        if self.merge_ratio == 1:
            return x, attention_mask

        if attention_mask is not None:
            if attention_mask.ndim == 4:
                b, _, l1, l2 = attention_mask.shape
                attention_mask = attention_mask.view(b, 1, self.merge_window_width, l1 // self.merge_window_width, 
                    self.merge_window_width, l2 // self.merge_window_width)

                attention_mask = torch.sum(attention_mask, dim=(2, 4), dtype=torch.bool)

                attention_mask = attention_mask.repeat_interleave(self.merge_window_width // self.merge_ratio, 2)
                attention_mask = attention_mask.repeat_interleave(self.merge_window_width // self.merge_ratio, 3)
            else:
                raise NotImplementedError(f'MLP patch merger is not implemented for attention_mask.ndim={attention_mask.ndim}')

        if x.shape[1] % self.merge_window_width != 0:
            x = F.pad(x, (0, 0, 0, self.merge_window_width - x.shape[1] % self.merge_window_width), value=0)

        x = x.view(x.shape[0], -1, self.merge_window_width, x.shape[-1])        
        x = x.view(x.shape[0], -1, self.merge_window_width * x.shape[-1])

        x = self.linear(x)
        x = x.view(x.shape[0], -1, x.shape[-1] * self.merge_ratio // self.merge_window_width)

        return x, attention_mask


class LinearPatchUnMerger(nn.Module):
    def __init__(self, merge_ratio, hidden_size, merge_window_width):
        super().__init__()

        self.hidden_size = hidden_size
        self.merge_ratio = merge_ratio
        self.merge_window_width = merge_window_width

        if self.merge_ratio != 1:
            self.linear = nn.Linear(hidden_size * merge_window_width // merge_ratio, hidden_size * merge_window_width)

    def forward(self, x):
        if self.merge_ratio == 1:
            return x

        x = x.view(x.shape[0], -1, x.shape[-1] * self.merge_window_width // self.merge_ratio)

        x = self.linear(x)

        x = x.view(x.shape[0], x.shape[1], self.merge_window_width, -1)
        x = x.view(x.shape[0], x.shape[1] * self.merge_window_width, -1)

        return x


class MLPPatchMerger(nn.Module):
    def __init__(self, merge_ratio, hidden_size, merge_window_width):
        super().__init__()

        self.hidden_size = hidden_size
        self.merge_ratio = merge_ratio
        self.merge_window_width = merge_window_width

        if self.merge_ratio != 1:
            self.mlp = nn.Sequential(
                nn.RMSNorm(hidden_size * merge_window_width),
                nn.Linear(hidden_size * merge_window_width, hidden_size * merge_window_width),
                nn.SiLU(),
                nn.Linear(hidden_size * merge_window_width, hidden_size * merge_window_width // merge_ratio)
            )

    def forward(self, x, attention_mask=None):
        if self.merge_ratio == 1:
            return x, attention_mask

        if attention_mask is not None:
            if attention_mask.ndim == 4:
                b, _, l1, l2 = attention_mask.shape
                attention_mask = attention_mask.view(b, 1, self.merge_window_width, l1 // self.merge_window_width, 
                    self.merge_window_width, l2 // self.merge_window_width)

                attention_mask = torch.sum(attention_mask, dim=(2, 4), dtype=torch.bool)

                attention_mask = attention_mask.repeat_interleave(self.merge_window_width // self.merge_ratio, 2)
                attention_mask = attention_mask.repeat_interleave(self.merge_window_width // self.merge_ratio, 3)
            else:
                raise NotImplementedError(f'MLP patch merger is not implemented for attention_mask.ndim={attention_mask.ndim}')

        if x.shape[1] % self.merge_window_width != 0:
            x = F.pad(x, (0, 0, 0, self.merge_window_width - x.shape[1] % self.merge_window_width), value=0)

        x = x.view(x.shape[0], -1, self.merge_window_width, x.shape[-1])        
        x = x.view(x.shape[0], -1, self.merge_window_width * x.shape[-1])

        x = self.mlp(x)
        x = x.view(x.shape[0], -1, x.shape[-1] * self.merge_ratio // self.merge_window_width)

        return x, attention_mask


class MLPPatchUnMerger(nn.Module):
    def __init__(self, merge_ratio, hidden_size, merge_window_width):
        super().__init__()

        self.hidden_size = hidden_size
        self.merge_ratio = merge_ratio
        self.merge_window_width = merge_window_width

        if self.merge_ratio != 1:
            self.mlp = nn.Sequential(
                nn.RMSNorm(hidden_size * merge_window_width // merge_ratio),
                nn.Linear(hidden_size * merge_window_width // merge_ratio, hidden_size * merge_window_width),
                nn.SiLU(),
                nn.Linear(hidden_size * merge_window_width, hidden_size * merge_window_width)
            )

    def forward(self, x):
        if self.merge_ratio == 1:
            return x

        x = x.view(x.shape[0], -1, x.shape[-1] * self.merge_window_width // self.merge_ratio)

        x = self.mlp(x)

        x = x.view(x.shape[0], x.shape[1], self.merge_window_width, -1)
        x = x.view(x.shape[0], x.shape[1] * self.merge_window_width, -1)

        return x


class SumPatchMerger(nn.Module):
    def __init__(self, merge_ratio, hidden_size, merge_window_width):
        super().__init__()

        self.hidden_size = hidden_size
        self.merge_ratio = merge_ratio
        self.merge_window_width = merge_window_width

        if merge_window_width > merge_ratio:
            print('Passing merge_window_width > merge_ratio to a SumPatchMerger wont have any effect')

    def forward(self, x, attention_mask=None):
        if self.merge_ratio == 1:
            return x, attention_mask

        if attention_mask is not None:
            if attention_mask.ndim == 4:
                b, _, l1, l2 = attention_mask.shape
                attention_mask = attention_mask.view(b, 1, self.merge_ratio, l1 // self.merge_ratio, 
                    self.merge_ratio, l2 // self.merge_ratio)

                attention_mask = torch.sum(attention_mask, dim=(2, 4), dtype=torch.bool)
            else:
                raise NotImplementedError(f'Sum patch merger is not implemented for attention_mask.ndim={attention_mask.ndim}')

        if x.shape[1] % self.merge_ratio != 0:
            x = F.pad(x, (0, 0, 0, self.merge_ratio - x.shape[1] % self.merge_ratio), value=0)

        x = x.view(x.shape[0], -1, self.merge_ratio, x.shape[-1]) 
        x = x.sum(dim=2)

        return x, attention_mask


class SumPatchUnMerger(nn.Module):
    def __init__(self, merge_ratio, hidden_size, merge_window_width):
        super().__init__()

        self.hidden_size = hidden_size
        self.merge_ratio = merge_ratio
        self.merge_window_width = merge_window_width

        if merge_window_width > merge_ratio:
            print('Passing merge_window_width > merge_ratio to a SumPatchMerger wont have any effect')

    def forward(self, x):
        if self.merge_ratio == 1:
            return x

        x = x.repeat_interleave(2, dim=1)

        return x


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


class PerceiverResamplerMerger(nn.Module):
    def __init__(self, merge_ratio, hidden_size, merge_window_width):
        super().__init__()

        self.hidden_size = hidden_size
        self.merge_ratio = merge_ratio
        self.merge_window_width = merge_window_width

        self.learnable_latents = nn.Parameter(torch.empty(1, merge_window_width // merge_ratio, hidden_size))
        self.local_attn = MultiHeadAttentionWithRoPE(hidden_size, 8)
        self.global_attn = MultiHeadAttentionWithRoPE(hidden_size, 8)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.learnable_latents, a=math.sqrt(5))


    def forward(self, x, attention_mask):
        # x: (B, L, E), where L might be more than merge_window_size
        # attention_mask: (B, 1, L, L) a mix of padding mask and causal mask
        # attention_mask shound be (B * L / l, 1, M, l) where M - number of learnable latents and l is merge_window_size

        # I had no opportunity to debug this, tho it looks correct
        B, L, H = x.shape

        if attention_mask is not None:
            attention_mask = torch.cat([attention_mask[..., i * self.merge_window_width: (i + 1) * self.merge_window_width, i * self.merge_window_width: (i + 1) * self.merge_window_width] 
                for i in range(L // self.merge_window_width + (L % self.merge_window_width != 0))])

            attention_mask = attention_mask[:, :, -1].unsqueeze(2).repeat(1, 1, self.merge_window_width // self.merge_ratio, 1)

        x = x.reshape(-1, self.merge_window_width, H)
        x = self.local_attn(self.learnable_latents.repeat(x.shape[0], 1, 1), x, x, attention_mask)

        x = x.reshape(B, L // self.merge_ratio, H)
        x = self.global_attn(x, x, x)

        attention_mask = torch.ones((B, 1, x.shape[1], x.shape[1]), dtype=torch.bool, device=x.device)
        return x, attention_mask
    