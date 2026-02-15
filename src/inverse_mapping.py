import torch

def inverse_reinhard(ldr, exposure=1.0):
    eps = 1e-6
    return (ldr / torch.clamp(1 - ldr, min=eps)) / exposure