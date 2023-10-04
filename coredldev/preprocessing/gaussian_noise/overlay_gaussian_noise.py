import torch
from ...utilites._preprocessing import power


def overlay_gaussian_noise(device="cpu"):
    def inner(tensor, params):
        tensor_power = power(tensor)
        noise = torch.normal(mean=0, std=1, size=tensor.shape).to(device)
        noise_power = power(noise)
        snr = params[-1]
        if snr == 0:
            return tensor
        noise = noise * (float(tensor_power) / float(noise_power)) / float(snr)
        return tensor.to(device) + noise.to(device)

    return inner
