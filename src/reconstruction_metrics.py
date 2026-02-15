import torch
from .perceptual_encoding import compute_luminance, pu_encode


def rmse(a, b):
    return torch.sqrt(torch.mean((a - b) ** 2))


def log_rmse(a, b):
    eps = 1e-6
    log_a = torch.log10(torch.clamp(a, min=eps))
    log_b = torch.log10(torch.clamp(b, min=eps))
    return torch.sqrt(torch.mean((log_a - log_b) ** 2))


def pu_error(a, b, peak=1000):
    """
    Compute perceptual error using PU-style encoding.
    """

    La = compute_luminance(a)
    Lb = compute_luminance(b)

    pu_a = pu_encode(La, peak_nits=peak)
    pu_b = pu_encode(Lb, peak_nits=peak)

    return torch.mean((pu_a - pu_b) ** 2)


def dynamic_range_error(a, b):
    eps = 1e-6

    La = compute_luminance(a).flatten()
    Lb = compute_luminance(b).flatten()

    qa_high = torch.quantile(La, 0.999)
    qb_high = torch.quantile(Lb, 0.999)

    return torch.abs(torch.log10((qa_high + eps) / (qb_high + eps)))
