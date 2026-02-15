import torch

def pu_encode(luminance, peak_nits=1000):
    L = torch.clamp(luminance / peak_nits, min=1e-6)
    return torch.log10(1 + 5000 * L) / torch.log10(torch.tensor(5001.0))

def compute_luminance(img):
    return 0.2126*img[...,0] + 0.7152*img[...,1] + 0.0722*img[...,2]