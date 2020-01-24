import h5py
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal
import scipy.interpolate
import pdb
from analyze_prc import create_pseudo_random_code, analyze_prc, create_estimation_matrix
in_fname = 'data/RRI_20190603_154238_154736_lv1_v5.h5'
sample_freq = 62500.
sample_freq_i = 50000.

clen = 1000
seed = 0
Nranges = 1000
rfi_rem = False
 
def main():
    data = load_epop(in_fname)

    # Downsample and chunk up the data
    chunkind = np.where(np.diff(data['packet']) > 1)
    ind0 = 0
    for ind1 in chunkind[0]:
        toc = tic()
        x_o = np.arange(ind0, ind1)
        I = data['I'][ind0:ind1]
        Q = data['I'][ind0:ind1]
        finind = np.isfinite(I)

        slen_i = sample_freq / sample_freq_i
        nsamples = np.sum(finind)
        nsamples_i = np.floor(nsamples / slen_i)
            
        x_i = np.arange(x_o[finind].min(), x_o[finind].max(), slen_i)
        
        I_i = scipy.interpolate.interp1d(x_o[finind], I[finind])(x_i)
        Q_i = scipy.interpolate.interp1d(x_o[finind], Q[finind])(x_i)
        #I_i = scipy.signal.resample(I[finind], int(nsamples_i))
        #Q_i = scipy.signal.resample(Q[finind], int(nsamples_i))
        sig = I_i + 1j * Q_i

        # Analyze signal for the waveform
        anlen = int(len(sig) / clen) * clen
        sig = sig[:anlen]
        N, B = calc_waveform(anlen, clen, seed, Nranges)
        spec = np.zeros([N, Nranges], dtype=np.complex64)
        res = np.zeros([N, Nranges], dtype=np.complex64)

        # print('NRanges: %i, ndop: %i' % (Nranges, anlen / clen))
        for i in np.arange(N):
            z = sig[i * clen:(i + 1) * clen]
            z = z - np.median(z)  # remove dc
            res[i, :] = np.dot(B, z)
        for i in np.arange(Nranges):
            # FFT on the convolved signal 
            # Gaussian pulse shaping reduces out-of-band emissions
            spec[:, i] = np.fft.fftshift(np.fft.fft(
                scipy.signal.blackmanharris(N) * res[:, i]
            ))
        if rfi_rem:
            median_spec = np.zeros(N, dtype=np.float32)
            for i in np.arange(N):
                median_spec[i] = np.median(np.abs(spec[i, :]))
            for i in np.arange(Nranges):
                spec[:, i] = spec[:, i] / median_spec[:]
        pwr = np.abs(spec)
        max_pwr = pwr.max()
        mi = np.where(pwr == max_pwr)
        print('RMax: %i km, Max: %1.1f x \n' % (np.mean(mi[1]) * 6, max_pwr / np.mean(pwr.flatten())))
        # print(toc())

        ind0 = ind1


def load_epop(in_fname, freq=6.4E6):
    data = {}
    f = h5py.File(in_fname, 'r')
    data['I'] = f['RRI Data']['Radio Data Monopole 1 (mV)'][...].flatten()
    data['Q'] = f['RRI Data']['Radio Data Monopole 2 (mV)'][...].flatten()
    freqs = f['RRI Data']['Channel A Frequencies (Hz)'][...].flatten()
    packets = f['RRI Data']['RRI Packet Numbers'][...]
    data['packet'] = np.tile(packets, (29, 1)).T.flatten()
    finind = np.isfinite(data['I'])
    finind_2 = np.isfinite(data['Q'])
    assert np.sum(finind) == np.sum(finind_2), 'indices must match'
    freqind = freqs == freq
    for k, v in data.items():
        data[k] = v[freqind]
    return data


def calc_waveform(an_len, clen, seed=0, Nranges=1000):
    code = create_pseudo_random_code(clen, seed)
    N = an_len / clen
    assert N == np.floor(N), 'N is not an integer'
    N = int(N)
    res = np.zeros([N, Nranges], dtype=np.complex64)
    r = create_estimation_matrix(code=code, rmax=Nranges)
    B = r['B']  # B is the estimation matrix
    return N, B

from time import time

def tic():
    """Simple helper function for timing executions.

    Returns a closure that holds current time when calling tic().

    Usage:
    toc = tic()
    //some code
    print(toc())
    """
    t = time()
    return lambda: (time() - t)



if __name__ == '__main__':
    main()
