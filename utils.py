import torch
from torch import nn
from torch.utils.flop_counter import FlopCounterMode

def get_flops(model: nn.Module, inp: torch.Tensor | tuple, with_backward=False):
    
    istrain = model.training
    model.eval()
    
    inp = inp if isinstance(inp, torch.Tensor) else torch.randn(inp)

    flop_counter = FlopCounterMode(display=False, depth=None)
    with flop_counter:
        if with_backward:
            out = model(inp)
            if isinstance(out, torch.Tensor):
                out.sum().backward()
            else:
                out[0].sum().backward()
        else:
            model(inp)
    total_flops =  flop_counter.get_total_flops()
    if istrain:
        model.train()
    return total_flops