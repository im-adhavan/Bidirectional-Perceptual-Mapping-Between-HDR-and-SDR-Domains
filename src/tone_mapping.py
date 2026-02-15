import torch

def reinhard_global(hdr, exposure=1.0):
    x = hdr * exposure
    return x / (1.0 + x)

def filmic(hdr):
    A,B,C,D,E,F = 0.15,0.50,0.10,0.20,0.02,0.30
    x = torch.clamp(hdr, min=0)
    return ((x*(A*x+C*B)+D*E)/(x*(A*x+B)+D*F)) - E/F