import matplotlib
import numpy, pylab, logging
import seaborn as sns
sns.set_context('talk') 
sns.set(font_scale=1)
sns.set_palette('colorblind')
sns.set(rc={'figure.figsize':(40,10)})
sns.set_style('ticks')

pylab.rcParams['axes.linewidth'] = 1

from coredldev.utilites.noise import generate_colored_noise, get_time_domain_strain
from lal import LIGOTimeGPS

def to_pycbc_timeseries(time_domain_strain, sampling_frequency=4096, start_time = 0):
    '''
    Convert to PyCBC TimeSeries. Easier to add signals.
    '''
    from pycbc.types.timeseries import TimeSeries
    return TimeSeries(time_domain_strain, 
                      epoch=LIGOTimeGPS(start_time),
                      delta_t=1./sampling_frequency)


def inject_signal(noise, 
                  parameters, 
                  approximant='IMRPhenomXAS', 
                  f_lower=5, f_ref=5, get_signal = False):
    logging.info('Simulating waveform with {} approximant'.format(approximant))
    
    from pycbc.waveform import get_td_waveform, taper_timeseries
    from pycbc.detector import Detector
    
    hplus, hcross = get_td_waveform(approximant=approximant,
                                    mass1=parameters['mass1'], 
                                    mass2=parameters['mass2'],
                                    spin1z=parameters['spin1z'], 
                                    spin2z=parameters['spin2z'],
                                    distance=parameters['distance'], 
                                    coa_phase=parameters['azimuth'],
                                    inclination=parameters['inclination'], 
                                    f_lower=f_lower,
                                    f_ref=f_ref, 
                                    delta_t=1.0/noise.sample_rate)
    
    detector = Detector('H1') 
    print(hplus.lal(), hcross.lal(),detector.lal())

    # hplus.start_time = LIGOTimeGPS(float(hplus.start_time) + parameters['tc'])
    # hcross.start_time = LIGOTimeGPS(float(hcross.start_time) + parameters['tc'])
    hplus = taper_timeseries(hplus, 'startend')
    hcross = taper_timeseries(hcross, 'startend')
    print(hplus.lal(), hcross.lal(),detector.lal())

    logging.info('Projecting signal to detector frame')
    signal = detector.project_wave(hplus, hcross,
                                   parameters['ra'], parameters['dec'], 
                                   parameters['psi'],
                                   method='lal',
                                   reference_time=parameters['tc'])
    
    if get_signal:
        logging.info('Returning signal')
        return signal 
    else:
        logging.info('Adding signal to noise')
        return noise.inject(signal)
    

def get_psd(strain,
            strain_high_pass = 5):
    '''
    Estimate PSD of PyCBC TimeSeries using Welch method
    '''
    from pycbc.psd import interpolate, inverse_spectrum_truncation, welch
    psd_estimation = 'median'
    psd_segment_length = 16
    psd_segment_stride = 8
    psd_inverse_length = 16
    psd_num_segments = 63
    psd_duration = 4
    psd_stride = 2
    
    logging.info('Estimating PSD')
    
    psd = welch(strain, avg_method=psd_estimation,
                seg_len=int(psd_segment_length * strain.sample_rate + 0.5),
                seg_stride=int(psd_segment_stride * strain.sample_rate + 0.5),
                num_segments=psd_num_segments,
                require_exact_data_fit=False)
    
    psd = interpolate(psd, 1. / strain.duration)
    psd = inverse_spectrum_truncation(psd,
                                      int(psd_inverse_length * strain.sample_rate),
                                      low_frequency_cutoff=strain_high_pass,
                                      trunc_method='hann')
    return psd

def whiten(strain,
           f_lower = 5.,):
    psd = get_psd(strain, strain_high_pass=f_lower)
    # Whiten the data by the asd correctly
    whiten = (4 * strain.to_frequencyseries() / (psd**0.5 * strain.get_duration())) .to_timeseries()    
    return whiten    

'''
The following example demonstrates how to run the script.
'''
start_time = 1242442967.
parameters = {'mass1' : 135,
              'mass2': 135,
              'spin1z' : 0.0,
              'spin2z': 0.0,
              'inclination': 2.9,
              'azimuth' : 0,
              'distance' : 5300,
              'ra' : 1.99,
              'dec' : -1.2,
              'psi' : 0,
              'tc' : 1242443167.}

frequency_domain_strain, frequencies = generate_colored_noise(psd_file='./aLIGO_O4_high_asd.txt', 
                                                              sampling_frequency=8192,duration = 4)
strain = get_time_domain_strain(frequency_domain_strain, 
                                sampling_frequency=8192)
print(len(strain))

noise = to_pycbc_timeseries(strain, start_time=1242442967)
signal = inject_signal(noise, parameters=parameters,get_signal=True) * 500
signal.start_time = 0
noise.start_time = 0
print(len(signal),signal.delta_t, len(noise), noise.delta_t)
sns.lineplot(signal.numpy(), label = "signal")
sns.lineplot(noise.inject(signal).numpy(), label = "injected")
sns.lineplot(noise.numpy(), label='noise_orig')
pylab.savefig('signal.png')
