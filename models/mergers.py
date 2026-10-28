import torch
from torch import nn


class LinearPatchMerger(nn.Module):
    def __init__(self, merge_ratio, hidden_size):
        super().__init__()

        self.hidden_size = hidden_size
        self.merge_ratio = merge_ratio

        if self.merge_ratio != 1:
            self.linear = nn.Linear(hidden_size*merge_ratio, hidden_size)

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
                raise NotImplementedError(f'Linear patch merger is not implemented for attention_mask.ndim={attention_mask.ndim}')

        if x.shape[1] % self.merge_ratio != 0:
            x = F.pad(x, (0, 0, 0, self.merge_ratio - x.shape[1] % self.merge_ratio), value=0)

        x = x.view(x.shape[0], -1, self.merge_ratio, x.shape[-1])        
        x = x.view(x.shape[0], -1, self.merge_ratio * x.shape[-1])

        x = self.linear(x)

        return x, attention_mask


class LinearPatchUnMerger(nn.Module):
    def __init__(self, merge_ratio, hidden_size):
        super().__init__()

        self.hidden_size = hidden_size
        self.merge_ratio = merge_ratio

        if self.merge_ratio != 1:
            self.linear = nn.Linear(hidden_size, hidden_size*merge_ratio)

    def forward(self, x):
        if self.merge_ratio == 1:
            return x

        x = self.linear(x)

        x = x.view(x.shape[0], x.shape[1], self.merge_ratio, -1)
        x = x.view(x.shape[0], x.shape[1] * self.merge_ratio, -1)

        return x


class MLPPatchMerger(nn.Module):
    def __init__(self, merge_ratio, hidden_size):
        super().__init__()

        self.hidden_size = hidden_size
        self.merge_ratio = merge_ratio

        if self.merge_ratio != 1:
            self.mlp = nn.Sequential(
                nn.RMSNorm(hidden_size * 2),
                nn.Linear(hidden_size * 2, hidden_size * 2),
                nn.SiLU(),
                nn.Linear(hidden_size * 2, hidden_size)
            )

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
                raise NotImplementedError(f'MLP patch merger is not implemented for attention_mask.ndim={attention_mask.ndim}')

        if x.shape[1] % self.merge_ratio != 0:
            x = F.pad(x, (0, 0, 0, self.merge_ratio - x.shape[1] % self.merge_ratio), value=0)

        x = x.view(x.shape[0], -1, self.merge_ratio, x.shape[-1])        
        x = x.view(x.shape[0], -1, self.merge_ratio * x.shape[-1])

        x = self.mlp(x)

        return x, attention_mask


class MLPPatchUnMerger(nn.Module):
    def __init__(self, merge_ratio, hidden_size):
        super().__init__()

        self.hidden_size = hidden_size
        self.merge_ratio = merge_ratio

        if self.merge_ratio != 1:
            self.mlp = nn.Sequential(
                nn.RMSNorm(hidden_size),
                nn.Linear(hidden_size, hidden_size * 2),
                nn.SiLU(),
                nn.Linear(hidden_size * 2, hidden_size * 2)
            )

    def forward(self, x):
        if self.merge_ratio == 1:
            return x

        x = self.mlp(x)

        x = x.view(x.shape[0], x.shape[1], self.merge_ratio, -1)
        x = x.view(x.shape[0], x.shape[1] * self.merge_ratio, -1)

        return x


class SumPatchMerger(nn.Module):
    def __init__(self, merge_ratio, hidden_size):
        super().__init__()

        self.hidden_size = hidden_size
        self.merge_ratio = merge_ratio

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
    def __init__(self, merge_ratio, hidden_size):
        super().__init__()

        self.hidden_size = hidden_size
        self.merge_ratio = merge_ratio

    def forward(self, x):
        if self.merge_ratio == 1:
            return x

        x = x.repeat_interleave(2, dim=1)

        return x
    