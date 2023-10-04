from ...utilites._preprocessing import power
from ...utilites._NoiseGenerator import NoiseGenerator
from ..._filepaths._filepaths import ligopsd_path
import numpy as np

noisegenerator = NoiseGenerator()
psd = np.loadtxt(ligopsd_path).T
color = NoiseGenerator.piecewise_logarithmic(psd[0], psd[1])


def custom_oneD_noise():
    def _custom_oneD_noise(data):
        waveform = data[0]
        params = data[2]
        sam_p = data[3]
        snr = params[5]
        noise = noisegenerator.generate(1e-4, len(waveform), colour=color)
        wfpower = power(waveform)
        if snr == 0:
            return waveform
        noise_power = power(noise)
        noise = noise * (float(wfpower) / float(noise_power)) / float(snr)
        return waveform + noise

    return custom_oneD_noise
