#!python
# ----------------------------------------------------------------------------
# Copyright (c) 2017 Massachusetts Institute of Technology (MIT)
# All rights reserved.
#
# Distributed under the terms of the BSD 3-clause license.
#
# The full license is in the LICENSE file, distributed with this software.
# ----------------------------------------------------------------------------
"""Script for analyzing pseudorandom-coded waveforms.

See the following paper for a description and application of the technique:

Vierinen, J., Chau, J. L., Pfeffer, N., Clahsen, M., and Stober, G.,
Coded continuous wave meteor radar, Atmos. Meas. Tech., 9, 829-839,
doi:10.5194/amt-9-829-2016, 2016.

"""
import datetime
import glob
import itertools
import math
import os
import time
from argparse import ArgumentParser

import numpy as np
import scipy.signal
import pandas

import digital_rf as drf
import pdb


def create_pseudo_random_code(clen=1000, seed=0):
    """
    seed is a way of reproducing the random code without
    having to store all actual codes. the seed can then
    act as a sort of station_id.

    """
    np.random.seed(seed)
    phases = np.array(
        np.exp(1.0j * 2.0 * math.pi * np.random.random(clen)),
        dtype=np.complex64,
    )
    return(phases)


def periodic_convolution_matrix(envelope, rmin=0, rmax=100):
    """
    we imply that the number of measurements is equal to the number of elements
    in code

    """
    L = len(envelope)
    ridx = np.arange(rmin, rmax)
    A = np.zeros([L, rmax-rmin], dtype=np.complex64)
    for i in np.arange(L):
        A[i, :] = envelope[(i-ridx) % L]
    result = {}
    result['A'] = A
    result['ridx'] = ridx
    return(result)


B_cache = 0
r_cache = 0
B_cached = False
def create_estimation_matrix(code, rmin=0, rmax=1000, cache=True):
    global B_cache
    global r_cache
    global B_cached

    if not cache or not B_cached:
        r_cache = periodic_convolution_matrix(
            envelope=code, rmin=rmin, rmax=rmax,
        )
        A = r_cache['A']
        Ah = np.transpose(np.conjugate(A))
        B_cache = np.dot(np.linalg.inv(np.dot(Ah, A)), Ah)
        r_cache['B'] = B_cache
        B_cached = True
        return(r_cache)
    else:
        return(r_cache)


def analyze_prc(
                dirn='', channel='hfrx', idx0=0, an_len=1000000, clen=10000, station=0,
                Nranges=1000, rfi_rem=True, cache=True,
                ):
    r"""Analyze pseudorandom code transmission for a block of data.

    idx0 = start idx
    an_len = analysis length
    clen = code length
    station = random seed for pseudorandom code
    cache = Do we cache (\conj(A^T)\*A)^{-1}\conj{A}^T for linear least squares
        solution (significant speedup)
    rfi_rem = Remove RFI (whiten noise).
    Nranges = number of range gates

    """
    if type(dirn) is str:
        g = drf.DigitalRFReader(dirn)
    else:
        g = dirn

    code = create_pseudo_random_code(clen=clen, seed=station)
    N = an_len / clen  # What is N? Number of waveform repetitions in the signal
    assert N == np.floor(N), 'N is not an integer'
    N = int(N)

    res = np.zeros([N, Nranges], dtype=np.complex64)
    r = create_estimation_matrix(code=code, cache=cache, rmax=Nranges)
    B = r['B']  # B is the estimation matrix?
    spec = np.zeros([N, Nranges], dtype=np.complex64)

    for i in np.arange(N):
        z = g.read_vector_c81d(idx0 + i * clen, clen, channel)  # z is the signal
        z = z - np.median(z)  # remove dc
        res[i, :] = np.dot(B, z)
    for i in np.arange(Nranges):
        spec[:, i] = np.fft.fftshift(np.fft.fft(
            scipy.signal.blackmanharris(N) * res[:, i]  # Gaussian pulse shaping reduces out-of-band emissions
        ))

    if rfi_rem:
        median_spec = np.zeros(N, dtype=np.float32)
        for i in np.arange(N):
            median_spec[i] = np.median(np.abs(spec[i, :]))
        for i in np.arange(Nranges):
            spec[:, i] = spec[:, i] / median_spec[:]
    ret = {}
    ret['res'] = res
    ret['spec'] = spec

    return(ret)


def read_log(logfile):
    print('Loading log file')
    with open(logfile, 'r') as f:
        times = []
        freqs = []
        samples = []
        for line in f:
            if line[:4] == 'Tune':
                continue
            else:
                vals = line.split()
                times.append(datetime.datetime.strptime(vals[0], '%Y/%m/%d-%H:%M:%S.%f'))
                freqs.append(float(vals[1]))
                samples.append(int(vals[2]))
    times = np.array(times)
    freqs = np.array(freqs)
    samples = np.array(samples)

    time = times[:-1] + (times[1:] - times[:-1]) / 2
    anlen = samples[1:] - samples[:-1]
    data = {
            'freq': freqs[:-1],
             'idx': samples[:-1],
           'anlen': samples[1:] - samples[:-1],
            }
    df = pandas.DataFrame(data, index=time)
    return df
     

if __name__ == '__main__':
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    desc = """Script for analyzing pseudorandom-coded waveforms.

    See the following paper for a description and application of the technique:

    Vierinen, J., Chau, J. L., Pfeffer, N., Clahsen, M., and Stober, G.,
    Coded continuous wave meteor radar, Atmos. Meas. Tech., 9, 829-839,
    doi:10.5194/amt-9-829-2016, 2016.

    """

    parser = ArgumentParser(description=desc)

    parser.add_argument(
        'datadir', help='''Data directory to analyze.''',
    )
    parser.add_argument(
        '-c', '--ch', default='hfrx',
        help='''Channel name of data to analyze. (default: %(default)s)'''
    )
    parser.add_argument(
        '-o', '--out', dest='outdir', default='{datadir}/prc_analysis',
        help='''Processed output directory. (default: %(default)s)''',
    )
    parser.add_argument(
        '-x', '--delete_old', action='store_true', default=False,
        help='''Delete existing processed files.''',
    )
    parser.add_argument(
        '-n', '--logfile', dest='logfile', type=str, default='freqstep.log',
        help='''Frequency sample log file produced by tx_chirp.py (default: %(default)s)''',
    )
    parser.add_argument(
        '-l', '--code_length', dest='codelen', type=int, default=10000,
        help='''Code length. (default: %(default)s)''',
    )
    parser.add_argument(
        '-s', '--station', type=int, default=0,
        help='''Station ID for code (seed). (default: %(default)s)''',
    )
    parser.add_argument(
        '-r', '--nranges', type=int, default=1000,
        help='''Number of range gates. (default: %(default)s)''',
    )

    op = parser.parse_args()

    op.datadir = os.path.abspath(op.datadir)

    # join outdir to datadir to allow for relative path, normalize
    op.outdir = os.path.abspath(op.outdir.format(datadir=op.datadir))
    if not os.path.isdir(op.outdir):
        os.makedirs(op.outdir)
    datpath = os.path.join(op.outdir, 'last.dat')
    if op.delete_old:
        for f in itertools.chain(
            glob.iglob(datpath),
            glob.iglob(os.path.join(op.outdir, '*.png')),
        ):
            os.remove(f)

    data = drf.DigitalRFReader(op.datadir)
    sr = data.get_properties(op.ch)['samples_per_second']
    b = data.get_bounds(op.ch)

    # Define indexing according to the frequency stepping log file
    op.logfile = time.strftime(os.path.join(os.path.join(op.datadir, op.ch), op.logfile))
    idx_data = read_log(op.logfile)
  
    for time, row in idx_data.iterrows():
        try:
            dsp_delay = 1775  #1780    # 7968
            idx = np.array(int(row['idx'])) + dsp_delay
            res = analyze_prc(
                data, channel=op.ch, idx0=idx, an_len=int(row['anlen']), clen=op.codelen,
                station=op.station, Nranges=op.nranges,
                cache=True, rfi_rem=True,
            )

            plt.clf()

            M = 10.0 * np.log10((np.abs(res['spec'])))

            # calculate plot parameters
            tx_freq = row['freq'] * 1E6
            sample_rate = sr
            code_len_bauds = op.codelen
            freq_dwell_time = row['anlen'] / sample_rate

            # Range characteristics (y-axis)
            sample_len_secs = 1 / sample_rate
            rangegate = sample_len_secs * 3e8
            ranges = np.arange(op.nranges) * rangegate

            # Doppler characteristics (x-axis)
            code_len_secs = sample_len_secs * code_len_bauds
            tx_wlen = 3E8 / tx_freq
            doppler_bandwidth_hz = sample_rate / code_len_bauds
            doppler_res_hz = doppler_bandwidth_hz / (freq_dwell_time / code_len_secs)
            doppler_res_ms = doppler_res_hz * tx_wlen / 2
            vels = (np.arange(M.shape[0]) - M.shape[0] / 2) * doppler_res_ms
            
            plt.pcolormesh(vels, ranges / 1E3, np.transpose(M), vmin=(np.median(M) - 1.0))
            plt.ylabel('range (km)')
            plt.xlabel('Doppler velocity (m/s)')
            clb = plt.colorbar()
            clb.set_label('Intensity / dB')

            #plt.pcolormesh(np.transpose(M), vmin=(np.median(M) - 1.0))
            timestr = time.strftime('%Y-%m-%d %H:%M:%S')
            plt.title('%s %f MHz' % (timestr, row['freq']))
            plt.savefig(os.path.join(
                op.outdir, 'spec-{0:06d}.png'.format(int(np.uint64(idx / sr))),
            ))
            print('%s' % timestr)

        except IOError:
            print('IOError, skipping.')
