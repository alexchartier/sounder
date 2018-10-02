#!python
# ----------------------------------------------------------------------------
# Copyright (c) 2018 Johns Hopkins APL
# All rights reserved.
#
# Distributed under the terms of the BSD 3-clause license.
#
# The full license is in the LICENSE file, distributed with this software.
# ----------------------------------------------------------------------------
"""Step through oscillator frequencies"""

import pdb
from datetime import datetime, timedelta
import numpy as np 
import time 
from gnuradio import uhd
import pytz
import digital_rf as drf



def step(
         usrp, op, ch_num=0, 
                   sleeptime=0.1, 
                   out_fname=None
    ):
    """ Step the USRP's oscillator through a list of frequencies """

    print('Starting freq_stepper')
    freq_list = set_freq_list()
    if out_fname:
        with open(out_fname, 'a') as f:
            f.write('Tune time (UT)   Freq (MHz)   Tune sample\n')

    # Check for GPS lock
    print('Checking for GPS lock...')
    while not usrp.get_mboard_sensor("gps_locked", 0).to_bool():
        print("waiting for gps lock...")
        time.sleep(5)
    print('.....locked')

    # Begin infinite transmission loop
    freq = 0
    while 1:

        # GPS time is necessary to sync operations between the transmitter and receiver
        gpstime = datetime.utcfromtimestamp(usrp.get_mboard_sensor("gps_time"))
        gpstime_next = gpstime + timedelta(seconds=1)
        gpstime_secs = gpstime.replace(tzinfo=pytz.utc) - drf.util.epoch

        # USRP time is necessary to know what sample number we shift frequencies at
        usrptime_secs = usrp.get_time_now().get_real_secs()  # This gets it from the USRP - the USRP time first needs to be set to GPS time
        usrptime_next = drf.util.epoch + timedelta(seconds=usrptime_secs + 1)

        # Change frequency each time we hit a new time in the list, otherwise hold the existing note
        if ((gpstime_next.second) in freq_list.keys()) and (freq != freq_list[gpstime_next.second]):
            freq = freq_list[gpstime_next.second]

            # Specify USRP tune time on the first exact sample after listed time
            print(usrptime_next.strftime('USRP tune time: %Y%b%d %H:%M:%S'))
            tune_time_secs = (usrptime_next - drf.util.epoch).total_seconds()

            # Calculate the samplerate
            try:  
                ch_samplerate_frac = op.ch_samplerates_frac[ch_num]
                ch_samplerate_ld = (
                    np.longdouble(ch_samplerate_frac.numerator)
                    / np.longdouble(ch_samplerate_frac.denominator)
                )
            except:
                ch_samplerate_ld = op.samplerate

            tune_time_rsamples = np.ceil(tune_time_secs * op.samplerate)
            tune_time_secs = tune_time_rsamples / op.samplerate

            gps_lock = usrp.get_mboard_sensor("gps_locked").to_bool()
            print('GPS lock status: %s' % gps_lock)
            # Optionally write out the shift samples of each frequency
            tune_sample = int(np.uint64(tune_time_secs * ch_samplerate_ld))
            if out_fname:
                with open(gpstime_next.strftime(out_fname), 'a') as f:
                    # f.write('GPS lock status: %s' % str(gps_lock))
                    f.write('%s %s %i\n' % (gpstime_next.strftime('%Y/%m/%d-%H:%M:%S.%f'), str(freq).rjust(4), tune_sample))
          
            usrp.set_command_time(
                                  uhd.time_spec(float(tune_time_secs)),
                                  uhd.ALL_MBOARDS,
            )

            # Tune to the next frequency in the list
            tune_res = usrp.set_center_freq(
                            uhd.tune_request(freq * 1E6, op.lo_offsets[ch_num], \
                                             args=uhd.device_addr(','.join(op.tune_args)),
                                            ),
                            ch_num,
            )

            usrp.clear_command_time(uhd.ALL_MBOARDS)
            gpstime = datetime.utcfromtimestamp(usrp.get_mboard_sensor("gps_time"))
            if op.verbose:
                if freq == np.min(freq_list.values()):
                    print('\n')
                print('Tuned to %s MHz at GPS time %s (sample %i)' % (str(freq).rjust(4), gpstime.strftime('%Y%b%d %H:%M:%S'), tune_sample))

        time.sleep(sleeptime)


def set_dev_time(usrp, timetype):
    """ Set the USRP's time based on GPS or NTP"""
    if timetype == 'GPS':
        while int(usrp.get_time_last_pps().get_real_secs()) != usrp.get_mboard_sensor("gps_time").to_int():
            print('USRP time %i, GPS time %i' % (int(usrp.get_time_last_pps().get_real_secs()), usrp.get_mboard_sensor("gps_time").to_int()))
            usrp.set_time_now(uhd.time_spec_t(usrp.get_mboard_sensor("gps_time").to_int() + 2), uhd.ALL_MBOARDS)
            time.sleep(1)
            print('USRP time %i, GPS time %i' % (int(usrp.get_time_last_pps().get_real_secs()), usrp.get_mboard_sensor("gps_time").to_int()))
    elif timetype == 'NTP':
        tt = time.time()
        usrp.set_time_now(uhd.time_spec(tt), uhd.ALL_MBOARDS)
        # wait for time registers to be in known state
        time.sleep(1)

        print('Time set using %s' % timetype)


def set_freq_list():
    # time, freq (MHz)
    # This could be set in a text file, but it's important to understand the function above when choosing times
    return {
             0: 3.0,
            10: 4.6,
            20: 4.7,
            30: 4.8,
            40: 4.9,
            50: 5.1,
           }

    """
    return {
            0: 2, 
            5: 3, 
            10: 4, 
            15: 5,
            20: 6,
            25: 7,
            30: 8,
            35: 9,
            40: 10,
            45: 11,
            50: 12,
            55: 13,
            }


    return {
             0: 2,
             1: 2.5,
             2: 3,
             3: 3.5,
             4: 4,
             5: 4.5,
             6: 5,
             7: 5.5,
             8: 6,
             9: 6.5,
            10: 7,
            11: 7.5,
            12: 8,
            13: 8.5,
            14: 9,
            15: 9.5,
            16: 10,
            17: 10.5,
            18: 11,
            19: 11.5,
            20: 12,
            21: 5,
           }
    """

