import numpy as np
from pycbc.detector import Detector
from pycbc.types import TimeSeries
from pycbc.waveform import taper_timeseries

tc = 1242443167

def detector_angle_mixing(detector_name = 'H1'):
    detector = Detector(detector_name)
    def inner(inp):
        hplus = TimeSeries(inp["hplus"], delta_t=inp["params"]["sam_p"], epoch = tc)
        hcross = TimeSeries(inp["hcross"], delta_t=inp["params"]["sam_p"], epoch = tc)
        ra, dec, pol = inp["params"]["angle"]
        hplus = taper_timeseries(hplus, 'startend')
        hcross = taper_timeseries(hcross, 'startend')
        # print(hplus.lal(), hcross.lal(),detector.lal())
        signal = detector.project_wave(hplus,hcross, ra, dec, pol, method = "lal", reference_time = tc)
        inp["signal"] = signal.numpy()
        del inp["hplus"]
        del inp["hcross"]
        
        return inp
    return inner