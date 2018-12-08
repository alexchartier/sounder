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

# inputs  (first three MUST match in Tx and Rx)
sample_rate = 5E5
baud_oversampling = 10  
code_len_bauds = 2000  # (-l)

nranges = 2000  # (-r)
freq_dwell_time = 5   # for chirpsounder the radar sits on a specified frequency until it moves to the next one
tx_freq = 10E6
freq_list = np.linspace(2, 15, 12) * 1E6

# Standard stuff
sample_size_bytes = 4
day_secs = 86400
terabyte_units = 1E12  # bytes

# Calculations

# First of all, print the inputs
an_len_samples = freq_dwell_time * sample_rate
tx_wlen = 3E8 / tx_freq
sample_len_secs = 1 / sample_rate
baud_len_secs = sample_len_secs * baud_oversampling 
code_len_samples = code_len_bauds * baud_oversampling
code_len_secs = code_len_bauds * baud_oversampling / sample_rate

print('\n\n*** specified inputs ***')
print("carrier_freq: %2.1f MHz (wavelength: %1.1f m)" % (tx_freq / 1E6, tx_wlen))
print("Sample rate: %2.1f MHz" % (sample_rate / 1e6))
print("Bauds: %i (%ix oversampled), baud length: %1.2f microseconds" % (code_len_bauds, baud_oversampling, baud_len_secs * 1E6))
print("Code length: %2.2f secs" % code_len_secs)
print("Ranges analyzed: %i" % nranges)
print('Analysis length: %1.1f seconds, %i samples ' % (freq_dwell_time, an_len_samples))
print('Frequency list: %s' % str(freq_list / 1E6))
print('Ne list: %s x 1E10 electrons/m3' % str(1.24 * (freq_list/1E6) ** 2)) 
print('Ne 30deg incidence list: %s x 1E10 electrons/m3' % str(1.24 * (freq_list/1E6) ** 2 / np.cos(np.deg2rad(60)))) 
print('\n\n*** expected outputs at %2.2f MHz (***' % (tx_freq / 1E6))

# How much velocity resolution to expect?
print('Transmitter bandwidth: %2.2f kHz\n' % (sample_rate / (code_len_bauds / baud_oversampling) / 1E3))
tx_wlen = 3E8 / tx_freq
doppler_bandwidth_hz = sample_rate / (baud_oversampling * code_len_bauds)
doppler_res_hz = doppler_bandwidth_hz / (freq_dwell_time / code_len_secs)
print('Doppler bandwidth: %2.2f Hz (%2.2f m/s) \nDoppler resolution: %2.2f Hz (%2.2f m/s)' \
	% (doppler_bandwidth_hz, doppler_bandwidth_hz * tx_wlen / 2, doppler_res_hz, doppler_res_hz * tx_wlen / 2))

# How much range resolution to expect?
sample_len_secs = 1 / sample_rate
baud_len_secs = sample_len_secs * baud_oversampling 
rangegate = baud_len_secs * 3e8
range_res = rangegate / 2  # Not sure why this is the case - see paper
print('\nrange aliasing occurs at %1.1f km' % (code_len_secs * 3E8 / 1E3))
print('rangegate size %1.1f km?' % (baud_len_secs * 3E5))
print('altitude resolution (1/2 rangegate) at %1.1f MHz sampling: %2.2f km' % (sample_rate / 1E6,  range_res / 1E3))


# How much data to expect?
tera_day = sample_rate * sample_size_bytes * day_secs / terabyte_units / baud_oversampling
tera_year = tera_day * 365
print("%2.2f terabytes/day, %2.2f terabytes/year for %2.1f MHz sampling (based on %i byte samples and %ix oversampling)" % (tera_day, tera_year, sample_rate / 1E6, sample_size_bytes, baud_oversampling))

print('\n\n')
