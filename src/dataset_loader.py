import os
import torch
import OpenEXR
import Imath
import numpy as np

def read_exr(path, device="cpu"):
    exr = OpenEXR.InputFile(path)
    dw = exr.header()["dataWindow"]
    w = dw.max.x - dw.min.x + 1
    h = dw.max.y - dw.min.y + 1
    FLOAT = Imath.PixelType(Imath.PixelType.FLOAT)

    R = np.frombuffer(exr.channel("R", FLOAT), dtype=np.float32).reshape(h, w)
    G = np.frombuffer(exr.channel("G", FLOAT), dtype=np.float32).reshape(h, w)
    B = np.frombuffer(exr.channel("B", FLOAT), dtype=np.float32).reshape(h, w)

    img = np.stack([R, G, B], axis=-1)
    return torch.from_numpy(img).to(device)

def stream_dataset(folder, device="cpu"):
    for f in os.listdir(folder):
        if f.endswith(".exr"):
            yield f.replace(".exr",""), read_exr(os.path.join(folder,f), device)