from ...utilites._preprocessing import power
from ...utilites._NoiseGenerator import NoiseGenerator
from ..._filepaths._filepaths import ligopsd_path
import numpy as np
import pycbc
from pycbc import noise as pycbc_noise

psd = pycbc.types.load_frequencyseries(ligopsd_path)
delta_t = 1.0 / (4096 * 2)


def noise(data):
    waveform = data[0]
    params = data[1]
    sam_p = data[2]
    snr = params[5]
    noise = np.array(pycbc_noise.noise_from_psd(len(waveform), sam_p/100, psd, seed=42))
    wfpower = power(waveform)
    if snr == 0:
        return waveform
    noise_power = power(noise)
    noise = noise * (float(wfpower) / float(noise_power)) / float(snr)
    return waveform + noise
