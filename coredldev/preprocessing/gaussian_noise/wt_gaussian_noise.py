import torch
from ...utilites._preprocessing import power, noisewt


def wt_gaussian_noise(device="cpu"):
    def inner(a, *args):
        tensor, params = a[0], a[1]
        tensor_power = power(tensor)
        noise = torch.normal(mean=0, std=1, size=tensor.shape[0]).to(device)
        noise_power = power(noise)
        snr = params[-1]
        if snr == 0:
            return tensor
        noise = noisewt(noise) * (float(tensor_power) / float(noise_power)) / float(snr)
        return tensor.to(device) + noise.to(device)

    return inner
