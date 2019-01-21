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
from __future__ import absolute_import, division, print_function

from datetime import datetime
import glob
import itertools
import math
import os
import time
from argparse import ArgumentParser

import digital_rf as drf
import numpy as np
import scipy.signal
from scipy.sparse import csr_matrix
import pdb

import fnmatch
import pickle
import sys 
sys.path.append('./waveforms/')
from freq_stepper import get_freq_list


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

    """
    if type(dirn) is str:
        g = drf.DigitalRFReader(dirn)
    else:
        g = dirn

    code = create_pseudo_random_code(clen=clen, seed=station)

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
        '-f', '--freq_list', dest='freq_list_fname', default='freq_list.txt',
        help='''Text file with list of tune times in format:
        time (in seconds of each minute): frequency (in MHz), e.g.:
        0:   3
        15:  6
        30:  9
        45:  12
        (default: %(default)s)''',
    )    
    parser.add_argument(
        '-l', '--code_length', dest='codelen', type=int, default=1000,
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
        help='''Produce range-doppler plots''',
    )
    parser.add_argument(
        '-q', '--save', action='store_true', default=True,
        help='''Save out spectra''',
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
    # join outdir to datadir to allow for relative path, normalize
    op.outdir = os.path.abspath(op.outdir.format(datadir=op.datadir))
    if not os.path.isdir(op.outdir):
        print('Making %s' % op.outdir)
        os.makedirs(op.outdir)

    # Define directories 
    plotdir = os.path.join(op.outdir, '%s/plots' % op.ch)
    savedir = os.path.join(op.outdir, '%s/spectra' % op.ch)

    #  Delete old if necessary
    datpath = os.path.join(op.outdir, '%s/last.dat' % op.ch)
    if op.delete_old:
        for root, dirnames, filenames in os.walk(op.outdir):
            for f in filenames:
                if f.endswith(('.png', '.pkl', 'last.dat')):
                    os.remove(os.path.join(root, f))

    # See where we got up to before.
    data = drf.DigitalRFReader(op.datadir)
    sr = data.get_properties(op.ch)['samples_per_second']
    b = data.get_bounds(op.ch)
    idx = np.array(b[0])
    if os.path.isfile(datpath):
        fidx = np.fromfile(datpath, dtype=np.int)
        if b[0] <= fidx:
            idx = fidx

    # Get frequency information from freq_list
    chdir = os.path.join(op.datadir, op.ch)
    flist = get_freq_list(os.path.join(chdir, op.freq_list_fname))
    freqs = flist.values()
    shift_secs = flist.keys()
    shift_secs = np.array(shift_secs)
    shift_secs.sort()
    shift_secs_ext = np.append(shift_secs, shift_secs[:2] + 60)
    
    # get samplerate from file (used to get timestamp) 
    srn = data.get_properties(op.ch, sample=idx)['sample_rate_numerator']
    srd = data.get_properties(op.ch, sample=idx)['sample_rate_denominator']
    sr = srn / srd
    """
    # set up directory
    dirn = os.path.join(savedir, op.ch)
    try:
        print('200 Making %s' % dirn)
        os.makedirs(dirn)
    except:
        None
    """
    # start processing
    while True:
        # move index forward if we are not on a tune time
        idx = np.ceil(idx / sr) * sr  # start by putting it on an integer second
        dtime = datetime.utcfromtimestamp(idx / sr)
        diff = shift_secs_ext - dtime.second
        diff[diff >= 60] -= 60
        min_diff = diff[diff >= 0].min()
        idx += min_diff * np.float(sr)
        dtime = datetime.utcfromtimestamp(idx / sr)
        diff = shift_secs - dtime.second
    
        idx = int(idx)

        # figure out analysis length based on frequency shifting
        shiftind = np.where(diff == 0)[0][0]
        nextind = shiftind + 1
        op.anlen = int((shift_secs_ext[nextind] - shift_secs[shiftind]) * sr)
        tune_freq = flist[shift_secs[shiftind]]

        # Wait if we don't have enough data
        if idx + op.anlen > b[1]:
            print('waiting for more data, sleeping.')
            time.sleep(op.anlen / sr)
            b = data.get_bounds(op.ch)
            continue

        # Process
        try:
            delay = 7650  # Should be 7650 for zero range offset
            res = analyze_prc(
                data, channel=op.ch, idx0=idx + delay, an_len=op.anlen, clen=op.codelen,
                station=op.station, Nranges=op.nranges,
                cache=True, rfi_rem=True,
            )

            pwr = np.abs(res['spec'])
            M = 10.0 * np.log10(pwr)
            # calculate plot parameters
            rg = 3e8 * np.arange(op.nranges) / sr / 1e3
            ndop = op.anlen / op.codelen
            dop_hz = np.fft.fftshift(np.fft.fftfreq(int(ndop), d=op.codelen / sr))
            dop_vel = (dop_hz / (tune_freq * 1E6)) * 3E8

            maxind = np.unravel_index(M.argmax(), M.shape)
            print('%i   Freq: %02.2f: Max. value: %02.2f dB at %2.1f m/s, %2.1f km'\
                 % (idx, tune_freq, M.max(), dop_vel[maxind[0]], rg[maxind[1]]))

            if op.plot:
                plt.clf()
                plt.pcolormesh(dop_vel, rg, np.transpose(M), vmin=(np.median(M) - 1.0),)# vmax=10.)
                plt.ylabel('range (km)')
                plt.xlabel('Doppler velocity (m/s)')
                clb = plt.colorbar()
                clb.set_label('Intensity / dB')

                timestr = dtime.strftime('%Y-%m-%d %H:%M:%S')
                plt.title('%s %f MHz' % (timestr, tune_freq))
                dirn = os.path.join(plotdir, dtime.strftime('%Y%m%d'))
                try:
                    os.makedirs(dirn)
                except:
                    None
                plt.savefig(os.path.join(
                    plotdir, 'spec-{0:06d}.png'.format(int(np.uint64(idx / sr))),
                ))
                print('%s' % timestr)

            if op.save:
                # Take weighted mean across Doppler bins for each range. 
                dop_vel_2d = np.tile(dop_vel, (M.shape[1], 1)).T 
                mean_dop, sum_wts = np.average(dop_vel_2d, weights=pwr, axis=0, returned=True)
                out = {
                    'dop_vel': np.squeeze(mean_dop),
                    'int_pwr': np.squeeze(sum_wts),
                    'max_pwr_db': M.max(axis=0),
                }

                # Save the strong signals
                sigind = np.any(M > op.threshold, axis=0)
                if np.any(sigind):
                    for k, v in out.items():
                        v[np.invert(sigind)] = 0
                        out[k] = csr_matrix(v) 
                    
                    dirn = os.path.join(savedir, dtime.strftime('%Y%m%d'))
                    try:
                        os.makedirs(dirn)
                    except:
                        None
                    spec_fname_t = dtime.strftime('%2.2f_MHz' % tune_freq + '_%H%M%S_.pkl')
                    out_fname = os.path.join(dirn, spec_fname_t) 
                    with open(out_fname, 'wb') as f:
                        print('Saving to %s' % out_fname)
                        pickle.dump(out, f)
    
                # Make metadata file
                try:
                    metadata = {
                        'doppler': dop_vel,
                        'range': rg,
                    }
                    meta_out_fname = os.path.join(savedir, 'meta_%2.2f_.pkl' % tune_freq) 
                    with open(meta_out_fname, 'wb') as f:
                        pickle.dump(metadata, f)
                except:
                    None


        except IOError:
            print('IOError, skipping.')
        idx = idx + op.anlen
        
        np.array(idx).tofile(datpath)
