#!python
# ----------------------------------------------------------------------------
# Copyright (c) 2017 Massachusetts Institute of Technology (MIT)
# All rights reserved.
#
# Distributed under the terms of the BSD 3-clause license.
#
# The full license is in the LICENSE file, distributed with this software.
# ----------------------------------------------------------------------------
"""Transmit waveforms with synchronized USRPs."""
from __future__ import print_function

import math
import os
import re
import sys
import time
from argparse import (
    Action, ArgumentParser, Namespace, RawDescriptionHelpFormatter,
)
from datetime import datetime, timedelta
from fractions import Fraction
from itertools import chain, cycle, islice, repeat
from subprocess import call
from textwrap import TextWrapper, dedent, fill

import numpy as np
import pytz
from gnuradio import analog, blocks, gr, uhd
from six.moves import configparser

import pdb
import digital_rf as drf
import freq_stepper


def evalint(s):
    """Evaluate string to an integer."""
    return int(eval(s, {}, {}))


def evalfloat(s):
    """Evaluate string to a float."""
    return float(eval(s, {}, {}))


def noneorstr(s):
    """Turn empty or 'none' string to None."""
    if s.lower() in ('', 'none'):
        return None
    else:
        return s


def noneorbool(s):
    """Turn empty or 'none' string to None, all others to boolean."""
    if s.lower() in ('', 'none'):
        return None
    elif s.lower() in ('true', 't', 'yes', 'y', '1'):
        return True
    else:
        return False


def noneorcomplex(s):
    """Turn empty or 'none' to None, else evaluate to complex."""
    if s.lower() in ('', 'none'):
        return None
    else:
        return complex(eval(s, {}, {}))


class Extend(Action):
    """Action to split comma-separated arguments and add to a list."""

    def __init__(self, option_strings, dest, type=None, **kwargs):
        if type is not None:
            itemtype = type
        else:
            def itemtype(s):
                return s

        def split_string_and_cast(s):
            return [itemtype(a.strip()) for a in s.strip().split(',')]

        super(Extend, self).__init__(
            option_strings, dest, type=split_string_and_cast, **kwargs
        )

    def __call__(self, parser, namespace, values, option_string=None):
        cur_list = getattr(namespace, self.dest, [])
        if cur_list is None:
            cur_list = []
        cur_list.extend(values)
        setattr(namespace, self.dest, cur_list)


def read_timing_mode_waveform(mode_ini, iq_dir=None):
    """Read waveforms in timing mode format.

    This class supports the loading of a timing mode. The mode configuration
    file specifies the needed IQ data for each transmit sweep in the mode
    sequence. This is then used to load and RF waveform IQ data.


    Parameters
    ----------

    mode_ini : str
        Path to mode INI file. The '.iq.ini' extension can optionally be left
        off.

    iq_dir : str
        Path to directory with binary IQ sweep files. If None, defaults to
        '`os.path.dirname(mode_ini)`/../iq'.


    Returns
    -------

    dict
        Dictionary with entries::

            waveform : numpy.complex64 array
                Waveform (sequence) array.

            samplerate : float
                Waveform sample rate in Hz.

            center_freq : float
                Default waveform center frequency in Hz.

    """
    # make sure mode_ini exists as named, or add extension
    mode_ini = os.path.abspath(mode_ini)
    if not os.path.isfile(mode_ini):
        mode_ini = mode_ini + '.iq.ini'

    # parse mode INI file (so we can fail early if it's bad)
    cparser = configparser.SafeConfigParser()
    try:
        cparser.read(mode_ini)
    except configparser.MissingSectionHeaderError:
        raise ValueError('Cannot read timing mode INI file.')
    if not cparser.has_section('mode'):
        raise ValueError('Cannot read timing mode INI file.')

    # get mode name and default iq directory if unspecified
    mode_dir = os.path.dirname(mode_ini)
    base_dir = os.path.dirname(mode_dir)
    if iq_dir is None:
        iq_dir = os.path.join(base_dir, 'iq')

    # extract waveform characteristics and default frequency from mode acronym
    acronym = cparser.get('mode', 'name')
    name_segs = acronym.split('_')
    # grp = name_segs[0]
    # type_letter = name_segs[1]
    # ipp_ms = int(name_segs[2])
    # pulse_len_ms = int(name_segs[3])
    if name_segs[4].startswith('f'):
        if len(name_segs) > 5 and name_segs[5].startswith('f'):
            # desc = 'chirp'
            f0 = 1e5*int(name_segs[4][1:])
            f1 = 1e5*int(name_segs[5][1:])
            center_freq = (f0 + f1)/2
        else:
            # desc = 'unc'
            center_freq = 1e5*int(name_segs[4][1:])
    else:
        # baud_length_us = int(name_segs[4])
        # desc = name_segs[5]
        center_freq = 1e5*int(name_segs[6][1:])

    # read sample rate from INI file
    chip_ns = int(cparser.get('mode', 'chip_length'))
    samplerate_frac = Fraction(1000000000, chip_ns)
    samplerate = float(samplerate_frac)

    # now get list of iq data files from the mode sweep string
    sweep_seq = cparser.get('mode', 'sweeps').strip('[]').split(',')
    sweep_files = [os.path.join(iq_dir, s + '.iq.dat') for s in sweep_seq]

    # read the waveforms from the binary IQ files, stored as interleaved 16-bit
    # integers
    # keep each as a separate array initially since we may want to introduce
    # an IPP between each
    waveforms_int = [np.fromfile(f, dtype=np.int16) for f in sweep_files]
    # cast to complex64 and scale integer range to [0, 1]
    waveforms = [(w_i[0::2] + 1j*w_i[1::2])/(2**15-1) for w_i in waveforms_int]

    # join waveform sequence into single waveform assuming no extra IPP
    waveform = np.concatenate(waveforms)

    return dict(
        waveform=waveform, center_freq=center_freq, samplerate=samplerate
    )


class Tx(object):
    """Transmit data in binary format from a single USRP."""

    def __init__(
        self, waveform, amplitudes=[0.25], phases=[0],
        mboards=[], subdevs=['A:0'],
        centerfreqs=[440e6], lo_offsets=[0],
        lo_sources=[''], lo_exports=[None],
        dc_offsets=[None], iq_balances=[None],
        gains=[0], bandwidths=[0], antennas=[''],
        samplerate=1e6,
        dev_args=[], stream_args=[], tune_args=[],
        sync=True, sync_source='gpsdo',
        realtime=False, verbose=True, test_settings=True,
        freq_list_fname=None,
    ):
        options = locals()
        del options['self']
        op = self._parse_options(**options)
        self.op = op

        if op.test_settings:
            # test usrp device settings, release device when done
            if op.verbose:
                print('Initialization: testing device settings.')
            usrp = self._usrp_setup()
            del usrp

    @staticmethod
    def _parse_options(**kwargs):
        """Put all keyword options in a namespace and normalize them."""
        op = Namespace(**kwargs)

        # determine mboard and subdev per channel, get number of channels
        op.mboards_bychan = []
        op.subdevs_bychan = []
        op.mboardnum_bychan = []
        mboards = op.mboards if op.mboards else ['default']
        for mbnum, (mb, sd) in enumerate(zip(mboards, op.subdevs)):
            sds = sd.split()
            mbs = list(repeat(mb, len(sds)))
            mbnums = list(repeat(mbnum, len(sds)))
            op.mboards_bychan.extend(mbs)
            op.subdevs_bychan.extend(sds)
            op.mboardnum_bychan.extend(mbnums)
        op.nmboards = len(op.mboards) if len(op.mboards) > 0 else 1
        op.nchs = len(op.mboards_bychan)

        # repeat arguments as necessary
        op.amplitudes = list(islice(cycle(op.amplitudes), 0, op.nchs))
        op.phases = list(islice(cycle(op.phases), 0, op.nchs))
        op.subdevs = list(islice(cycle(op.subdevs), 0, op.nmboards))
        op.centerfreqs = list(islice(cycle(op.centerfreqs), 0, op.nchs))
        op.dc_offsets = list(islice(cycle(op.dc_offsets), 0, op.nchs))
        op.iq_balances = list(islice(cycle(op.iq_balances), 0, op.nchs))
        op.lo_offsets = list(islice(cycle(op.lo_offsets), 0, op.nchs))
        op.lo_sources = list(islice(cycle(op.lo_sources), 0, op.nchs))
        op.lo_exports = list(islice(cycle(op.lo_exports), 0, op.nchs))
        op.gains = list(islice(cycle(op.gains), 0, op.nchs))
        op.bandwidths = list(islice(cycle(op.bandwidths), 0, op.nchs))
        op.antennas = list(islice(cycle(op.antennas), 0, op.nchs))

        # create device_addr string to identify the requested device(s)
        op.mboard_strs = []
        for n, mb in enumerate(op.mboards):
            if re.match(r'[^0-9]+=.+', mb):
                idtype, mb = mb.split('=')
            elif re.match(
                r'[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}', mb
            ):
                idtype = 'addr'
            elif (
                re.match(r'usrp[123]', mb) or re.match(r'b2[01]0', mb)
                or re.match(r'x3[01]0', mb)
            ):
                idtype = 'type'
            elif re.match(r'[0-9A-Fa-f]{1,}', mb):
                idtype = 'serial'
            else:
                idtype = 'name'
            if len(op.mboards) == 1:
                # do not use identifier numbering if only using one mainboard
                s = '{type}={mb}'.format(type=idtype, mb=mb.strip())
            else:
                s = '{type}{n}={mb}'.format(type=idtype, n=n, mb=mb.strip())
            op.mboard_strs.append(s)

        if op.verbose:
            opstr = dedent('''\
                Main boards: {mboard_strs}
                Subdevices: {subdevs}
                Frequency: {centerfreqs}
                LO frequency offset: {lo_offsets}
                LO source: {lo_sources}
                LO export: {lo_exports}
                DC offset: {dc_offsets}
                IQ balance: {iq_balances}
                Amplitude: {amplitudes}
                Phases: {phases}
                Gain: {gains}
                Bandwidth: {bandwidths}
                Antenna: {antennas}
                Device arguments: {dev_args}
                Stream arguments: {stream_args}
                Tune arguments: {tune_args}
                Sample rate: {samplerate}
            ''').strip().format(**op.__dict__)
            print(opstr)

        # check that subdevice specifications are unique per-mainboard
        for sd in op.subdevs:
            sds = sd.split()
            if len(set(sds)) != len(sds):
                errstr = (
                    'Invalid subdevice specification: "{0}". '
                    'Each subdevice specification for a given mainboard must '
                    'be unique.'
                )
                raise ValueError(errstr.format(sd))

        return op

    def _usrp_setup(self):
        """Create, set up, and return USRP sink object."""
        op = self.op
        # create usrp sink block
        usrp = uhd.usrp_sink(
               device_addr=','.join(chain(op.mboard_strs, op.dev_args)),
               stream_args=uhd.stream_args(
                    cpu_format='fc32',
                    otw_format='sc16',
                    channels=range(op.nchs),
                    args=','.join(op.stream_args)
               )
        )

        # set clock and time source if synced
        if op.sync:
            try:
                usrp.set_clock_source(op.sync_source, uhd.ALL_MBOARDS)
                usrp.set_time_source(op.sync_source, uhd.ALL_MBOARDS)
            except RuntimeError:
                errstr = (
                    "Unknown sync_source option: '{0}'. Must be one of {1}."
                ).format(op.sync_source, usrp.get_clock_sources(0))
                raise ValueError(errstr)

        # check for ref lock
        mbnums_with_ref = [
            mb_num for mb_num in range(op.nmboards)
            if 'ref_locked' in usrp.get_mboard_sensor_names(mb_num)
        ]
        if mbnums_with_ref:
            if op.verbose:
                sys.stdout.write('Waiting for reference lock...')
                sys.stdout.flush()
            timeout = 0
            while not all(
                usrp.get_mboard_sensor('ref_locked', mb_num).to_bool()
                for mb_num in mbnums_with_ref
            ):
                if op.verbose:
                    sys.stdout.write('.')
                    sys.stdout.flush()
                time.sleep(1)
                timeout += 1
                if timeout > 30:
                    if op.verbose:
                        sys.stdout.write('failed\n')
                        sys.stdout.flush()
                    raise RuntimeError(
                        'Failed to lock to 10 MHz reference.'
                    )
            if op.verbose:
                sys.stdout.write('locked\n')
                sys.stdout.flush()

        # set mainboard options
        for mb_num in range(op.nmboards):
            usrp.set_subdev_spec(op.subdevs[mb_num], mb_num)
        # set global options
        # sample rate
        usrp.set_samp_rate(float(op.samplerate))
        # read back actual value
        samplerate = usrp.get_samp_rate()
        # calculate longdouble precision sample rate
        # (integer division of clock rate)
        cr = usrp.get_clock_rate()
        srdec = int(round(cr / samplerate))
        samplerate_ld = np.longdouble(cr) / srdec
        op.samplerate = samplerate_ld
        sr_rat = Fraction(cr).limit_denominator() / srdec
        op.samplerate_num = sr_rat.numerator
        op.samplerate_den = sr_rat.denominator

        # Set device time using the onboard GPS
        freq_stepper.set_dev_time(usrp)

        # set per-channel options
        # set command time so settings are synced
        gpstime = datetime.utcfromtimestamp(usrp.get_mboard_sensor("gps_time"))
        gpstime_secs = (pytz.utc.localize(gpstime) - drf.util.epoch).total_seconds()
        COMMAND_DELAY = 0.2
        cmd_time_secs = gpstime_secs + COMMAND_DELAY

        usrp.set_command_time(
            uhd.time_spec(float(cmd_time_secs)),
            uhd.ALL_MBOARDS,
        )

        for ch_num in range(op.nchs):
            # local oscillator sharing settings
            lo_source = op.lo_sources[ch_num]
            if lo_source:
                try:
                    usrp.set_lo_source(lo_source, uhd.ALL_LOS, ch_num)
                except RuntimeError:
                    errstr = (
                        "Unknown LO source option: '{0}'. Must be one of {1},"
                        " or it may not be possible to set the LO source on"
                        " this daughterboard."
                    ).format(lo_source, usrp.get_lo_sources(uhd.ALL_LOS, ch_num))
                    raise ValueError(errstr)
            lo_export = op.lo_exports[ch_num]
            if lo_export is not None:
                if not lo_source:
                    errstr = (
                        'Channel {0}: must set an LO source in order to set'
                        ' LO export.'
                    ).format(ch_num)
                    raise ValueError(errstr)
                usrp.set_lo_export_enabled(lo_export, uhd.ALL_LOS, ch_num)
            # center frequency and tuning offset
            tune_res = usrp.set_center_freq(
                uhd.tune_request(
                    op.centerfreqs[ch_num], op.lo_offsets[ch_num],
                    args=uhd.device_addr(','.join(op.tune_args)),
                ),
                ch_num,
            )
            # store actual values from tune result
            op.centerfreqs[ch_num] = (
                tune_res.actual_rf_freq + tune_res.actual_dsp_freq
            )
            op.lo_offsets[ch_num] = -tune_res.actual_dsp_freq
            # dc offset
            dc_offset = op.dc_offsets[ch_num]
            if dc_offset is not None:
                usrp.set_dc_offset(dc_offset, ch_num)
            # iq balance
            iq_balance = op.iq_balances[ch_num]
            if iq_balance is not None:
                usrp.set_iq_balance(iq_balance, ch_num)
            # gain
            usrp.set_gain(op.gains[ch_num], ch_num)
            # bandwidth
            bw = op.bandwidths[ch_num]
            if bw:
                usrp.set_bandwidth(bw, ch_num)
            # antenna
            ant = op.antennas[ch_num]
            if ant:
                try:
                    usrp.set_antenna(ant, ch_num)
                except RuntimeError:
                    errstr = (
                        "Unknown RX antenna option: '{0}'. Must be one of {1}."
                    ).format(ant, usrp.get_antennas(ch_num))
                    raise ValueError(errstr)

        # commands are done, clear time
        usrp.clear_command_time(uhd.ALL_MBOARDS)
        time.sleep(COMMAND_DELAY)

        # read back actual channel settings
        for ch_num in range(op.nchs):
            if op.lo_sources[ch_num]:
                op.lo_sources[ch_num] = usrp.get_lo_source(uhd.ALL_LOS, ch_num)
            if op.lo_exports[ch_num] is not None:
                op.lo_exports[ch_num] = usrp.get_lo_export_enabled(
                    uhd.ALL_LOS, ch_num,
                )
            op.gains[ch_num] = usrp.get_gain(ch_num)
            op.bandwidths[ch_num] = usrp.get_bandwidth(chan=ch_num)
            op.antennas[ch_num] = usrp.get_antenna(chan=ch_num)

        if op.verbose:
            print('Using the following devices:')
            chinfostrs = [
                'Motherboard: {mb_id} ({mb_addr}) | Daughterboard: {db_name}',
                'Subdev: {sub} | Antenna: {ant} | Gain: {gain} | Rate: {sr}',
                'Frequency: {freq:.3f} ({lo_off:+.3f}) | Bandwidth: {bw}',
            ]
            if any(op.lo_sources) or any(op.lo_exports):
                chinfostrs.append(
                    'LO source: {lo_source} | LO export: {lo_export}'
                )
            chinfo = '\n'.join(['  ' + l for l in chinfostrs])
            for ch_num in range(op.nchs):
                header = '---- {0} '.format(ch_num)
                header += '-' * (78 - len(header))
                print(header)
                usrpinfo = dict(usrp.get_usrp_info(chan=ch_num))
                info = {}
                info['mb_id'] = usrpinfo['mboard_id']
                mba = op.mboards_bychan[ch_num]
                if mba == 'default':
                    mba = usrpinfo['mboard_serial']
                info['mb_addr'] = mba
                info['db_name'] = usrpinfo['tx_subdev_name']
                info['sub'] = op.subdevs_bychan[ch_num]
                info['ant'] = op.antennas[ch_num]
                info['bw'] = op.bandwidths[ch_num]
                info['dc_offset'] = op.dc_offsets[ch_num]
                info['freq'] = op.centerfreqs[ch_num]
                info['gain'] = op.gains[ch_num]
                info['iq_balance'] = op.iq_balances[ch_num]
                info['lo_off'] = op.lo_offsets[ch_num]
                info['lo_source'] = op.lo_sources[ch_num]
                info['lo_export'] = op.lo_exports[ch_num]
                info['sr'] = op.samplerate
                print(chinfo.format(**info))
                print('-' * 78)

        return usrp

    def run(self):
        op = self.op

        # window in seconds that we allow for setup time so that we don't
        # issue a start command that's in the past when the flowgraph starts
        SETUP_TIME = 10

        # print current time and NTP status
        if op.verbose and sys.platform.startswith('linux'):
            try:
                call(('timedatectl', 'status'))
            except OSError:
                # no timedatectl command, ignore
                pass
        
        # get UHD USRP source
        usrp = self._usrp_setup()

        # set launch time
        # (at least 1 second out so USRP start time can be set properly and
        #  there is time to set up flowgraph)
        ltts = usrp.get_time_last_pps().get_real_secs() + 2
        usrp.set_start_time(uhd.time_spec(ltts))

        # populate flowgraph one channel at a time
        fg = gr.top_block()
        for k in range(op.nchs):
            mult_k = op.amplitudes[k] * np.exp(1j * op.phases[k])
            if op.waveform is not None:
                waveform_k = mult_k * op.waveform
                src_k = blocks.vector_source_c(
                    waveform_k.tolist(), repeat=True,
                )
            else:
                src_k = analog.sig_source_c(
                    0, analog.GR_CONST_WAVE, 0, 0, mult_k,
                )
            fg.connect(src_k, (usrp, k))

        # start the flowgraph once we are near the launch time
        # (start too soon and device buffers might not yet be flushed)
        # (start too late and device might not be able to start in time)

        while (ltts - usrp.get_mboard_sensor("gps_time").to_int()) > 1.2:
            time.sleep(0.1)
        fg.start()

        # Step the USRP through a list of frequencies
        basedir ='/'.join(op.freq_list_fname.split('/')[:-2]) 
        flog_fname = os.path.join(basedir, 'logs/freqstep.log')
        lock_fname = os.path.join(basedir, 'logs/gps_lock.log')

        freq_stepper.step(
            usrp, op, 
            freq_list_fname=op.freq_list_fname,
            flog_fname=flog_fname,
            lock_fname=lock_fname,
        ) 

        # wait until flowgraph stops
        try:
            fg.wait()
        except KeyboardInterrupt:
            # catch keyboard interrupt and simply exit
            pass
        fg.stop()
        # need to wait for the flowgraph to clean up, otherwise it won't exit
        fg.wait()
        print('done')
        sys.stdout.flush()


if __name__ == '__main__':
    scriptname = os.path.basename(sys.argv[0])

    formatter = RawDescriptionHelpFormatter(scriptname)
    width = formatter._width

    title = 'tx.py'
    copyright = 'Copyright (c) 2017 Massachusetts Institute of Technology'
    shortdesc = 'Transmit a waveform on a loop using synchronized USRPs.'
    desc = '\n'.join((
        '*'*width,
        '*{0:^{1}}*'.format(title, width-2),
        '*{0:^{1}}*'.format(copyright, width-2),
        '*{0:^{1}}*'.format('', width-2),
        '*{0:^{1}}*'.format(shortdesc, width-2),
        '*'*width,
    ))

    usage = (
        '%(prog)s [-m MBOARD] [-d SUBDEV] [-c CH] [-y ANT] [-f freq_list]'
        ' [-F OFFSET] \\\n'
        '{0:8}[-g GAIN] [-b BANDWIDTH] [-r RATE] [options]'
        ' FILE\n'.format('')
    )

    epi_pars = [
        '''\
        Arguments in the "mainboard" and "channel" groups accept multiple
        values, allowing multiple mainboards and channels to be specified.
        Multiple arguments can be provided by repeating the argument flag, by
        passing a comma-separated list of values, or both. Within each argument
        group, parameters will be grouped in the order in which they are given
        to form the complete set of parameters for each mainboard/channel. For
        any argument with fewer values given than the number of
        mainboards/channels, its values will be extended by repeatedly cycling
        through the values given up to the needed number.
        ''',
        '''\
        Arguments in other groups apply to all mainboards/channels (including
        the sample rate).
        ''',
        '''\
        Example usage:
        ''',
    ]
    epi_pars = [fill(dedent(s), width) for s in epi_pars]

    egtw = TextWrapper(
        width=(width - 2), break_long_words=False, break_on_hyphens=False,
        subsequent_indent=' ' * (len(scriptname) + 1),
    )
    egs = [
        '''\
        {0} -m 192.168.10.2 -d "A:0" -f freq_list.txt -F 12.5e6 -G 0.25 -g 0 -r 1e6
        code.bin
        ''',
    ]
    egs = [' \\\n'.join(egtw.wrap(dedent(s.format(scriptname)))) for s in egs]
    epi = '\n' + '\n\n'.join(epi_pars + egs) + '\n'

    # parse options
    parser = ArgumentParser(
        description=desc, usage=usage, epilog=epi,
        formatter_class=RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        '-q', '--quiet', dest='verbose', action='store_false',
        help='''Reduce text output to the screen. (default: False)''',
    )
    parser.add_argument(
        '--notest', dest='test_settings', action='store_false',
        help='''Do not test USRP settings until experiment start.
                (default: False)''',
    )

    wavgroup = parser.add_argument_group(title='waveform')
    wavgroup.add_argument(
        'file', nargs='?', default=None,
        help='''INI file specifying the waveform timing mode or complex64
                binary file giving the waveform directly.''',
    )
    wavgroup.add_argument(
        '--iq_dir', default=None,
        help='''Directory with int16 binary IQ sweep files corresponding to the
                mode INIT given with `file`. If None, defaults to
                "`os.path.dirname(file)`/../iq"''',
    )
    wavgroup.add_argument(
        '--tone', action='store_true',
        help='''Ignore other waveform arguments and output a tone at the
                specified frequency.''',
    )

    mbgroup = parser.add_argument_group(title='mainboard')
    mbgroup.add_argument(
        '-m', '--mainboard', dest='mboards', action=Extend,
        help='''Mainboard address. (default: first device found)''',
    )
    mbgroup.add_argument(
        '-d', '--subdevice', dest='subdevs', action=Extend,
        help='''USRP subdevice string. (default: "A:0")''',
    )

    chgroup = parser.add_argument_group(title='channel')
    chgroup.add_argument(
        '-F', '--lo_offset', dest='lo_offsets', action=Extend, type=float,
        help='''Frontend tuner offset from center frequency, in Hz.
                (default: 0)''',
    )
    chgroup.add_argument(
        '--lo_source', dest='lo_sources', action=Extend, type=noneorstr,
        help='''Local oscillator source. Typically 'None'/'' (do not set),
                'internal' (e.g. LO1 for CH1, LO2 for CH2),
                'companion' (e.g. LO2 for CH1, LO1 for CH2), or
                'external' (neighboring board via connector).
                (default: '')''',
    )
    chgroup.add_argument(
        '--lo_export', dest='lo_exports', action=Extend, type=noneorbool,
        help='''Whether to export the LO's source to the external connector.
                Can be 'None'/'' to skip the channel, otherwise it can be
                'True' or 'False' provided the LO source is set.
                (default: None)''',
    )
    chgroup.add_argument(
        '--dc_offset', dest='dc_offsets', action=Extend, type=noneorcomplex,
        help='''DC offset correction to use. Can be 'None'/'' to keep device
                default or a complex value (e.g. "1+1j"). (default: 0)''',
    )
    chgroup.add_argument(
        '--iq_balance', dest='iq_balances', action=Extend, type=noneorcomplex,
        help='''IQ balance correction to use. Can be 'None'/'' to keep device
                default or a complex value (e.g. "1+1j"). (default: 0)''',
    )
    chgroup.add_argument(
        '-G', '--amplitude', dest='amplitudes', action=Extend, type=float,
        help='''Waveform amplitude multiplier. (default: 0.25)''',
    )
    chgroup.add_argument(
        '-P', '--phase', dest='phases', action=Extend, type=float,
        help='''Waveform phase in degrees. (default: 0)''',
    )
    chgroup.add_argument(
        '-g', '--gain', dest='gains', action=Extend, type=float,
        help='''USRP frontend gain in dB. (default: 0)''',
    )
    chgroup.add_argument(
        '-b', '--bandwidth', dest='bandwidths', action=Extend, type=float,
        help='''Frontend bandwidth in Hz. (default: 0 == frontend default)''',
    )
    chgroup.add_argument(
        '-y', '--antenna', dest='antennas', action=Extend, type=noneorstr,
        help='''Name of antenna to select on the frontend.
                (default: 'None' == frontend default))''',
    )

    txgroup = parser.add_argument_group(title='transmitter')
    txgroup.add_argument(
        '-r', '--samplerate', dest='samplerate', type=evalfloat,
        help='''Sample rate in Hz. (default: waveform default or 1e6)''',
    )
    txgroup.add_argument(
        '-A', '--devargs', dest='dev_args', action=Extend,
        help='''Device arguments, e.g. "send_buff_size=1000000".
                (default: '')''',
    )
    txgroup.add_argument(
        '-a', '--streamargs', dest='stream_args', action=Extend,
        help='''Stream arguments, e.g. "fullscale=1.0".
                (default: '')''',
    )
    txgroup.add_argument(
        '-T', '--tuneargs', dest='tune_args', action=Extend,
        help='''Tune request arguments, e.g. "mode_n=integer,int_n_step=100e3".
                (default: '')''',
    )
    txgroup.add_argument(
        '--sync_source', dest='sync_source',
        help='''Clock and time source for all mainboards.
                (default: 'external')''',
    )
    txgroup.add_argument(
        '--nosync', dest='sync', action='store_false',
        help='''No syncing with external clock. (default: False)''',
    )

    timegroup = parser.add_argument_group(title='time')
    
    timegroup.add_argument(
        '-f', '--freq_list', dest='freq_list_fname',
        help='''Text file with list of tune times in format:
                time (in seconds of each minute): frequency (in MHz), e.g.:
                     0:  3
                    15:  6
                    30:  9
                    45:  12
                (default: None)''',
    ) 
    timegroup.add_argument(
        '-l', '--freq_log', dest='flog_fname',
        help='''Log file of tune times (stored in logs)''',
    )
    timegroup.add_argument(
        '-s', '--gps_log', dest='lock_fname',
        help='''Log of GPS lock status (stored in logs)''',
    )

    op = parser.parse_args()

    if not op.tone and op.file is None:
        raise ValueError('Must specify a waveform file or use "--tone".')

    # remove redundant arguments in dev_args, stream_args, tune_args
    if op.dev_args is not None:
        try:
            dev_args_dict = dict([a.split('=') for a in op.dev_args])
        except ValueError:
            raise ValueError(
                'Device arguments must be {KEY}={VALUE} pairs.'
            )
        op.dev_args = [
            '{0}={1}'.format(k, v) for k, v in dev_args_dict.iteritems()
        ]
    if op.stream_args is not None:
        try:
            stream_args_dict = dict([a.split('=') for a in op.stream_args])
        except ValueError:
            raise ValueError(
                'Stream arguments must be {KEY}={VALUE} pairs.'
            )
        op.stream_args = [
            '{0}={1}'.format(k, v) for k, v in stream_args_dict.iteritems()
        ]
    if op.tune_args is not None:
        try:
            tune_args_dict = dict([a.split('=') for a in op.tune_args])
        except ValueError:
            raise ValueError(
                'Tune request arguments must be {KEY}={VALUE} pairs.'
            )
        op.tune_args = [
            '{0}={1}'.format(k, v) for k, v in tune_args_dict.iteritems()
        ]

    options = {k: v for k, v in op._get_kwargs() if v is not None}
    fpath = options.pop('file')
    iq_dir = options.pop('iq_dir', None)
    tone = options.pop('tone', False)

    # read waveform
    if op.tone:
        options['waveform'] = None
    else:
        try:
            tm_dict = read_timing_mode_waveform(fpath, iq_dir=iq_dir)
            # use tm_dict as default options, overriding with command line
            options['waveform'] = tm_dict['waveform']
            options.setdefault('samplerate', tm_dict['samplerate'])
            options.setdefault('centerfreqs', [tm_dict['center_freq']])
        except ValueError:
            options['waveform'] = np.fromfile(op.file, dtype=np.complex64)

    tx = Tx(**options)
    tx.run()
