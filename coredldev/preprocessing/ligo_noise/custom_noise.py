from ...utilites._NoiseGenerator import NoiseGenerator
from ...utilites._preprocessing import power, noisewt
from ..._filepaths._filepaths import ligopsd_path
import torch
import numpy as np

noisegenerator = NoiseGenerator()
psd = np.loadtxt(ligopsd_path).T
color = NoiseGenerator.piecewise_logarithmic(psd[0], psd[1])


def custom_noise(device="cpu"):
    def inner(a, *args):
        tensor, params = a[0], a[1]
        tensor_power = params[3]
        snr = params[-1]
        noise = noisegenerator.generate(1e-4, 400, colour=color)
        noise_power = power(noise)
        if snr == 0:
            return tensor
        noise = noisewt(noise) * (float(tensor_power) / float(noise_power)) / float(snr)
        return tensor.to(device) + torch.tensor(noise).to(device)

    return inner
