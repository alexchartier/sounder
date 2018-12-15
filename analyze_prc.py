#!python
# ----------------------------------------------------------------------------
# Copyright (c) 2017 Massachusetts Institute of Technology (MIT)
# All rights reserved.
# 
# Modified by Alex T. Chartier, Johns Hopkins APL
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
from __future__ import absolute_import, division, print_function

from datetime import datetime
import glob
import itertools
import math
import os
import shutil
import time
from argparse import ArgumentParser

import digital_rf as drf
import numpy as np
import scipy.signal
import pdb

import sys 
sys.path.append('./waveforms/')
from freq_stepper import get_freq_list


def create_pseudo_random_code(clen=10000, seed=0):
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

    """
    if type(dirn) is str:
        g = drf.DigitalRFReader(dirn)
    else:
        g = dirn

    code = create_pseudo_random_code(clen=clen, seed=station)

    #Experimental
    # code  = waveform(station=station, clen=clen, filter_output=True)

    # <<<<<<

    N = an_len / clen
    assert N == np.floor(N), 'N is not an integer'
    N = int(N)
    res = np.zeros([N, Nranges], dtype=np.complex64)
    r = create_estimation_matrix(code=code, cache=cache, rmax=Nranges)
    B = r['B']  # B is the estimation matrix
    spec = np.zeros([N, Nranges], dtype=np.complex64)

    for i in np.arange(N):
        z = g.read_vector_c81d(idx0 + i * clen, clen, channel)  # z is the signal
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
    ret = {}
    ret['res'] = res
    ret['spec'] = spec
    return(ret)


def sort_freqs(chdir, fdir, freq_list_fname, ):
    """
    Move odin output into frequency-specific directories ahead of processing

    """
    print('moving output into frequency-specific directories')

    # Get frequency information from freq_list
    flist = get_freq_list(freq_list_fname)
    freqs = flist.values()
    shift_secs = flist.keys()
    shift_secs = np.array(shift_secs)
    shift_secs.sort()

    # Make frequency sub-directories
    freq_subdirs = {}
    for freq in freqs:
        f_subdir = os.path.join(os.path.join(chdir, fdir), str(freq))
        freq_subdirs[freq] = f_subdir
        try:
            os.makedirs(f_subdir)
        except:
            None
    shutil.copy2(freq_list_fname, f_subdir)

    # Move the files into frequency-specific subdirs
    subdirs = next(os.walk(chdir))[1]
    for subdir in subdirs:
        if (subdir != 'metadata') and (subdir != fdir):
            subdirn = os.path.join(chdir, subdir)
            files = os.listdir(subdirn)
            time.sleep(1)  # wait a few seconds so as not to move open files
            for f in files:
                try:
                    ts = datetime.utcfromtimestamp(float(f.split('.')[0][3:]))
                    diff = ts.second - shift_secs
                    shift_t = shift_secs[diff == diff[diff >= 0].min()]
                    assert len(shift_t) == 1, 'length is wrong'
                    shift_freq = flist[shift_t[0]]
                    shutil.move(
                        os.path.join(subdirn, f), \
                        os.path.join(freq_subdirs[shift_freq], subdirn.split('/')[-1]),
                    )
                except:
                    None

    # Copy over the metadata files
    for freq_subdir in freq_subdirs.values():
        shutil.copy(os.path.join(chdir, 'drf_properties.h5'), freq_subdir)
        new_metadatadir = os.path.join(freq_subdir, 'metadata')
        if os.path.exists(new_metadatadir):
            shutil.rmtree(new_metadatadir)
        shutil.copytree(os.path.join(chdir, 'metadata'), new_metadatadir)

    # Remove empty directories
    for subdir in subdirs:
        subdirn = os.path.join(chdir, subdir)
        try:
            os.rmdir(subdirn)
        except:
            None

    return freqs


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
        '-f', '--freq_list', dest='freq_list_fname',
        help='''Text file with list of tune times in format:
        time (in seconds of each minute): frequency (in MHz), e.g.:
        0:   3
        15:  6
        30:  9
        45:  12
        (default: None)''',
    )    
    parser.add_argument(
        '-l', '--code_length', dest='codelen', type=int, default=10000,
        help='''Code length. (default: %(default)s)''',
    )
    parser.add_argument(
        '-n', '--analysis_length', dest='anlen', type=int, default=6000000,
        help='''Analysis length. (default: %(default)s)''',
    )
    parser.add_argument(
        '-o', '--out', dest='outdir', default='{datadir}/prc_analysis',
        help='''Processed output directory. (default: %(default)s)''',
    )
    parser.add_argument(
        '-p', '--plot', action='store_true', default=False,
        help='''Produce plots instead of saving out spectra''',
    )
    parser.add_argument(
        '-r', '--nranges', type=int, default=1000,
        help='''Number of range gates. (default: %(default)s)''',
    )
    parser.add_argument(
        '-s', '--station', type=int, default=0,
        help='''Station ID for code (seed). (default: %(default)s)''',
    )
    parser.add_argument(
        '-t', '--threshold', type=float, default=7.0,
        help='''dB threshold for saving out spectra. (default: %(default)s)''',
    )
    parser.add_argument(
        '-x', '--delete_old', action='store_true', default=False,
        help='''Delete existing processed files.''',
    )

    op = parser.parse_args()

    op.datadir = os.path.abspath(op.datadir)
    

    # start processing
    while True:
        """   find a new way of doing this 
        if idx + op.anlen > b[1]:
            print('waiting for more data, sleeping.')
            time.sleep(op.anlen / sr)
            b = d.get_bounds(op.ch)
            continue
        """

        # Call sort_freqs
        chdir = os.path.join(op.datadir, op.ch)
        fdir = '_freqs'
        freqs = sort_freqs(chdir, fdir, op.freq_list_fname)

        # Loop over freqs
        for freq in freqs:
            freq = str(freq)
            tldir = os.path.join(chdir, fdir)
            freqdir = os.path.join(tldir, freq)
            print('Processing %s' % freqdir)

            # join outdir to datadir to allow for relative path, normalize
            op.outdir = os.path.abspath(op.outdir.format(datadir=freqdir))
            if not os.path.isdir(op.outdir):
                os.makedirs(op.outdir)
            datpath = os.path.join(op.outdir, 'last.dat')
            
            # Load data
            data = drf.DigitalRFReader(tldir)
            sr = data.get_properties(freq)['samples_per_second']
            b = data.get_bounds(freq)
            idx = np.array(b[0])
            if os.path.isfile(datpath):
                fidx = np.fromfile(datpath, dtype=np.int)
                if b[0] <= fidx:
                    idx = fidx

            try:
                os.makedirs(os.path.join(freqdir, 'spectra'))
            except:
                None

            # Process
            try:
                delay = 7650  # Should be 7650 for zero range offset
                pdb.set_trace()
                res = analyze_prc(
                    data, channel=op.ch, idx0=idx + delay, an_len=op.anlen, clen=op.codelen,
                    station=op.station, Nranges=op.nranges,
                    cache=True, rfi_rem=False,
                )
                plt.clf()

                M = 10.0 * np.log10((np.abs(res['spec'])))

                # calculate plot parameters
                tx_freq = row['freq'] * 1E6
                sample_rate = sr
                code_len_bauds = op.codelen
                freq_dwell_time = row['anlen'] / sample_rate

                maxind = np.unravel_index(M.argmax(), M.shape)
                print('Freq: %2.2f, Shape of M: %i x %i (Doppler x Range): Max. value: %2.2f dB at %i, %i'\
                     % (row['freq'], M.shape[0], M.shape[1], M.max(), maxind[0], maxind[1]))

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
               
                if op.plot: 
                    plt.clf()
                    plt.pcolormesh(vels, ranges / 1E3, np.transpose(M), vmin=(np.median(M) - 1.0),)# vmax=10.)
                    plt.ylabel('range (km)')
                    plt.xlabel('Doppler velocity (m/s)')
                    clb = plt.colorbar()
                    clb.set_label('Intensity / dB')

                    timestr = time.strftime('%Y-%m-%d %H:%M:%S')
                    plt.title('%s %f MHz' % (timestr, row['freq']))
                    plt.savefig(os.path.join(
                        op.outdir, 'spec-{0:06d}.png'.format(int(np.uint64(idx / sr))),
                    ))
                    print('%s' % timestr)

                else:
                    pdb.set_trace()
                    M[M < op.threshold] = 0
                    M = csparse(M)
                    spec_fname_t = time.strftime(spec_fname)
                    with open(spec_fname_t, 'wb') as f:
                        pickle.dump(M, f)

            except IOError:
                print('IOError, skipping.')
            print('%d' % (idx))
            idx = idx + op.anlen
            idx.tofile(datpath)
