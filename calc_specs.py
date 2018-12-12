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
code_len_bauds = 1000  # (-l)
sample_rate = 5E4

freq_dwell_time = 4   # for chirpsounder the radar sits on a specified frequency until it moves to the next one
freq_list = np.array([2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5,  6, 6.5, 7, 7.5, 8, 8.5, 9, 9.5, 10, 10.5, 11, 12, 13, 14, 16, 18])
freq_list = freq_list[::2]
freq_list *= 1E6

# Standard stuff
sample_size_bytes = 4
day_secs = 86400
terabyte_units = 1E12  # bytes

# Calculations

# First of all, print the inputs
an_len_samples = freq_dwell_time * sample_rate
sample_len_secs = 1 / sample_rate
code_len_secs = code_len_bauds  / sample_rate

np.set_printoptions(precision=3)
print('\n\n*** specified inputs ***')
print("carrier_freq (MHz)")
print(freq_list / 1E6)
print('Ne list (1E11 electrons/m3): %s ' % str(1.24 * (freq_list/1E6) ** 2 / 10)) 
print('Ne 30deg incidence (1E11 electrons/m3) list: %s' % str(1.24 * (freq_list/1E6) ** 2 / 10 * np.cos(np.deg2rad(60)))) 
print("wavelength (m)") 
print(3E8 / freq_list)
print("Sample rate: %2.2f kHz" % (sample_rate / 1e3))
print("Bauds: %i" % (code_len_bauds))
print("Code length: %2.3f secs" % code_len_secs)
print('Analysis length: %1.1f seconds, %i samples ' % (freq_dwell_time, an_len_samples))

# How much velocity resolution to expect?
print('Transmitter bandwidth: %2.2f kHz\n' % (sample_rate / 1E3))
tx_wlen = 3E8 / freq_list
doppler_bandwidth_hz = sample_rate / code_len_bauds
doppler_res_hz = doppler_bandwidth_hz / (freq_dwell_time / code_len_secs)
print('Doppler bandwidth: %2.2f Hz' % doppler_bandwidth_hz)
print('Doppler bandwidth (m/s)')
print(doppler_bandwidth_hz * tx_wlen / 2)
print('Doppler resolution (m/s)')
print(doppler_res_hz * tx_wlen / 2)

# How much range resolution to expect?
sample_len_secs = 1 / sample_rate
rangegate = sample_len_secs * 3e8
alt_res = rangegate / 2  
print('\nrange aliasing occurs at %1.1f km' % (code_len_secs * 3E8 / 1E3))
print('rangegate size %2.2f km' % (rangegate /1E3))

# How much data to expect?
tera_day = sample_rate * sample_size_bytes * day_secs / terabyte_units 
tera_year = tera_day * 365
print("%2.2f terabytes/day, %2.2f terabytes/year for %2.1f kHz sampling (based on %i byte samples)" \
        % (tera_day, tera_year, sample_rate / 1E3, sample_size_bytes))

print('\n\n')
