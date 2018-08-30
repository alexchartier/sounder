import pdb
import numpy as np

"""
# Juha paper inputs
sample_rate = 1E6
baud_oversampling = 10
code_len_bauds = 1000  # (-l)
nranges = 1000  # (-r)
chirp_len_secs = 2   # for chirpsounder the chirp is a defined length, resulting in a fixed analysis length
tx_freq = 33E6
"""

# inputs
sample_rate = 2E5
baud_oversampling = 10
code_len_bauds = 10000  # (-l)
nranges = 1000  # (-r)
chirp_len_secs = 2   # for chirpsounder the chirp is a defined length, resulting in a fixed analysis length
tx_freq = 3E6




an_len_samples = chirp_len_secs * sample_rate

sample_size_bytes = 8
day_secs = 86400
terabyte_units = 1E12  # bytes

tx_wlen = 3E8 / tx_freq
sample_len_secs = 1 / sample_rate
baud_len_secs = sample_len_secs * baud_oversampling 
code_len_samples = code_len_bauds * baud_oversampling

print('\n\n*** specified inputs ***')
print("carrier_freq: %2.1f MHz (wavelength: %1.1f m)" % (tx_freq / 1E6, tx_wlen))
print("Sample rate: %2.1f MHz" % (sample_rate / 1e6))
print("Bauds: %i (%ix oversampled), length: %1.2f microseconds" % (code_len_bauds, baud_oversampling, baud_len_secs * 1E6))
print("Ranges analyzed: %i" % nranges)
print('Analysis length: %1.1f seconds, %i samples ' % (chirp_len_secs, an_len_samples))

print('\n\n*** expected outputs ***')
# How much data to expect?
tera_day = sample_rate * sample_size_bytes * day_secs / terabyte_units
tera_year = tera_day * 365
print("%2.2f terabytes/day, %2.2f terabytes/year for %2.1f MHz sampling" % (tera_day, tera_year, sample_rate / 1E6))

# How much range resolution to expect?
rangegate = baud_len_secs * 3e8
range_res = rangegate / 2  # Not sure why this is the case - see paper
print('rangegate size %1.1f km?' % (baud_len_secs * 3E5))
print('range resolution (1/2 rangegate) at %1.1f MHz sampling: %2.2f km' % (sample_rate / 1E6,  range_res / 1E3))
print('range aliasing occurs at %1.1f km\n' % (rangegate * nranges / 1E3))
sample_spacing = baud_len_secs * code_len_bauds
print('Code length: %i bauds, %i samples, %1.3f seconds' % (code_len_bauds, code_len_samples, sample_spacing))

# How many Doppler bins to expect?
nbins = np.int(an_len_samples / code_len_samples)
doppler_vels = np.fft.fftfreq(nbins, d=sample_spacing) * tx_wlen
doppler_res_ms = doppler_vels[1] - doppler_vels[0]
doppler_bandwidth_ms = doppler_vels.max() - doppler_vels.min() + doppler_res_ms
doppler_res_hz = doppler_res_ms / 3E8

print('%i doppler bins, %f sample spacing' % (nbins, sample_spacing))
print('Doppler bandwidth %2.2f m/s at %1.2f m/s or %2.2E Hz resolution' % (doppler_bandwidth_ms, doppler_res_ms, doppler_res_hz))


print('\n\n')
