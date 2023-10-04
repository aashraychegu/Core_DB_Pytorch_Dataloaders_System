from ...utilites.noise import generate_colored_noise, get_time_domain_strain, get_frequency_domain_strain, to_pycbc_timeseries
from ..._filepaths._filepaths import ligopsd_path
def noise_injection():

    def inner(inp):
        signal = inp["signal"]
        delta_t = inp["pm_time"][-1] - inp["pm_time"][0]
        sam_p = inp["params"]["sam_p"]
        print(sam_p, 1/sam_p, delta_t)
        fnoise = generate_colored_noise(psd_file=ligopsd_path, sampling_frequency=1/sam_p,duration = len(signal)*sam_p)
        tds =  get_time_domain_strain(fnoise[0],sampling_frequency=1/sam_p)
        tds = to_pycbc_timeseries(tds, sampling_frequency=1/sam_p, start_time = 0)
        signal = to_pycbc_timeseries(signal, sampling_frequency=1/sam_p, start_time = 0)
        print(tds.delta_t,len(tds),signal.delta_t,len(signal))
        inp["signal"] = tds[0:len(signal)].inject(signal)
        return inp
    return inner

