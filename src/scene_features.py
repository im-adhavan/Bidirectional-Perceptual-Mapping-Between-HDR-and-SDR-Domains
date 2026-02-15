import torch
from src.perceptual_encoding import compute_luminance


def extract_scene_features(hdr):

    L = compute_luminance(hdr)

    L = L.flatten()

    q_low = torch.quantile(L, 0.001)
    q_high = torch.quantile(L, 0.999)
    dynamic_range = torch.log10(q_high / (q_low + 1e-8)).item()

    log_std = torch.std(torch.log10(L + 1e-8)).item()

    q_95 = torch.quantile(L, 0.95)
    highlight_ratio = torch.mean((L > q_95).float()).item()

    q_05 = torch.quantile(L, 0.05)
    shadow_ratio = torch.mean((L < q_05).float()).item()

    return {
        "dynamic_range": dynamic_range,
        "log_std": log_std,
        "highlight_ratio": highlight_ratio,
        "shadow_ratio": shadow_ratio
    }