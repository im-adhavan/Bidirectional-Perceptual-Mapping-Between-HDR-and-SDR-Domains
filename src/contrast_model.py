import torch

def suprathreshold_contrast(luminance):
    logL = torch.log10(torch.clamp(luminance, min=1e-6))
    return torch.std(logL)