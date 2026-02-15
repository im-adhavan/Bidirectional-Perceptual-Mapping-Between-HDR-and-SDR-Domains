import torch

def rgb_to_xyz(img):
    M = torch.tensor([[0.4124,0.3576,0.1805],
                      [0.2126,0.7152,0.0722],
                      [0.0193,0.1192,0.9505]], device=img.device)
    flat = img.reshape(-1,3)
    xyz = flat @ M.T
    return xyz.reshape(img.shape)

def chromaticity_error(a,b):
    xyz_a = rgb_to_xyz(a)
    xyz_b = rgb_to_xyz(b)

    xa = xyz_a[...,0] / torch.clamp(torch.sum(xyz_a,-1),1e-6)
    ya = xyz_a[...,1] / torch.clamp(torch.sum(xyz_a,-1),1e-6)

    xb = xyz_b[...,0] / torch.clamp(torch.sum(xyz_b,-1),1e-6)
    yb = xyz_b[...,1] / torch.clamp(torch.sum(xyz_b,-1),1e-6)

    return torch.mean((xa-xb)**2 + (ya-yb)**2)