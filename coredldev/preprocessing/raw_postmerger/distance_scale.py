import numpy as np
def distance_scale():
    def _distance_scale(data):
        signal = data["signal"].astype(np.float64)
        radii = data["params"]["current_extraction_radius"]
        scale_to_radii = data["params"]["rescale_to_radii"]
        # print(radii, scale_to_radii,((radii/scale_to_radii)**2))
        # print(np.mean(data["signal"]))
        signal = ((radii/scale_to_radii)**2) * signal
        # print(np.mean(signal))
        data["signal"] = signal
        return data
    return _distance_scale
