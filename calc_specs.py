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
sample_rate = 5E4  # 5E4


freq_dwell_time = 1   # for chirpsounder the radar sits on a specified frequency until it moves to the next one
#freq_list = np.array([2, 3.2, 4.4, 5.6, 6.8, 8., 9.2, 10.4, 11.6, 12.8, 14., 15.2]) * 1E6
dene_F = np.arange(0.4, 12.5, 0.4) * 1E11
freq_list =  9 * np.sqrt(dene_F / np.cos(np.deg2rad(30)))
"""
dene_F_min = (freq_list_E.max() / 9) ** 2 * np.cos(np.deg2rad(60))
dene_F = np.arange(dene_F_min, 1E12, 0.5E11)
freq_list_F = 9 * np.sqrt(dene_F / np.cos(np.deg2rad(60)))
"""
# freq_list = np.linspace(np.sqrt(1.8), np.sqrt(12.7), 30) ** 2

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
# print(freq_list / 1E6)
flist = ['%1.1f, ' % (f / 1E6) for f in freq_list]
print(''.join(flist))
print('Ne list (1E11 electrons/m3): %s ' % str(1.24 * (freq_list/1E6) ** 2 / 10)) 
print('Ne 60deg incidence (1E11 electrons/m3) list: %s' % str(1.24 * (freq_list/1E6) ** 2 / 10 * np.cos(np.deg2rad(30)))) 
print('Ne 30deg incidence (1E11 electrons/m3) list: %s' % str(1.24 * (freq_list/1E6) ** 2 / 10 * np.cos(np.deg2rad(60)))) 
print('Ne 10deg incidence (1E11 electrons/m3) list: %s' % str(1.24 * (freq_list/1E6) ** 2 / 10 * np.cos(np.deg2rad(80)))) 
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
rangegate = sample_len_secs * 3e8 / 1E3

from plot_rtd import calc_vht
rmax = code_len_secs * 3E8  / 1E3
print('\nrange aliasing occurs at %1.1f km' % (rmax))
print('rangegate size %2.2f km' % (rangegate))
vht = calc_vht(np.arange(0, rmax, rangegate))
print('Virtual heights')
print('%s' % str(vht[np.isfinite(vht)]))

# How much data to expect?
tera_day = sample_rate * sample_size_bytes * day_secs / terabyte_units 
tera_year = tera_day * 365
print("%2.2f terabytes/day, %2.2f terabytes/year for %2.1f kHz sampling (based on %i byte samples)" \
        % (tera_day, tera_year, sample_rate / 1E3, sample_size_bytes))

print('\n\n')
