#!/usr/bin/env python

import h5py

import matplotlib.pyplot as plt
import numpy as np
import sys, os
sys.path.append(os.path.expanduser('~/digital_rf-2.6.0b1/examples/sounder/'))
import prc_analyze as prc
import stuffr
import pdb

fname = "/home/alex/data/loopback_l1000_b10/ch0/2018-10-15T16-00-00/rf@1539619268.000.h5"
h=h5py.File(fname,"r")
z=h["rf_data"].value

code=prc.create_pseudo_random_code(clen=1000, seed=0)
print(code)

t1 = np.fft.fft(z)
t2 = np.fft.fft(np.conj(code),len(z))
t3 = np.ones(t1.shape, dtype=np.complex_) * np.nan
for ind, a in enumerate(t1):
    t3[ind] = a * t2[ind]
    
cc = np.abs(np.fft.ifft(t3))
# plot cross-correlation between code and measurement.
plt.plot(cc[0:20000])
plt.show()
