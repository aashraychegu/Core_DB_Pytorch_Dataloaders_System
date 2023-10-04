from ...utilites._preprocessing import pad_width, wt, window
from ...utilites._NoiseGenerator import NoiseGenerator
from ..._filepaths._filepaths import ligopsd_path
import numpy as np

def wavelet_transform():
    def _wavelet_transform(x):
        data = x["signal"]
        params = x["params"]
        sam_p = params["sam_p"]
        shift = params["percent_shift"]
        raw_wt = wt(window(data), sam_p=sam_p)
        print(raw_wt.shape)
        raw_wt = raw_wt[:,:int(raw_wt.shape[1]/2)]
        raw_wt = raw_wt[:,::50]
        x["signal"] = pad_width(raw_wt, percent_shift=shift)
        # x["signal"] = raw_wt
        return x

    return _wavelet_transform
