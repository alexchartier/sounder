sample_rate = 2E5
baud_oversampling = 10
code_len = 1000  # (-l)
nranges = 1000  # (-r)
an_len = 10000

sample_size_bytes = 8
day_secs = 86400
terabyte_units = 1E12

#Report inputs
print("Sample rate: %2.1f MHz" % (sample_rate / 1e6))
print("Bauds: %i (%ix oversampled)" % (code_len, baud_oversampling))
print("Ranges analyzed: %i" % nranges)

# How much data to expect?
tera_day = sample_rate * sample_size_bytes * day_secs / terabyte_units
tera_year = tera_day * 365
print("%2.2f terabytes/day, %2.2f terabytes/year for %2.2E sampling" % (tera_day, tera_year, sample_rate))

# How much range resolution to expect?
sample_len = 1 / sample_rate
baud_len = sample_len * baud_oversampling 
rangegate = (baud_len * 3e8) / 2
print('range resolution at %1.1f MHz sampling: %2.2f km' % (sample_rate / 1E6,  rangegate / 1E3))
print('aliasing occurs at %1.1f km' % (rangegate * nranges / 1E3))

# How many Doppler bins to expect?

