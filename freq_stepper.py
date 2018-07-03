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

def step(usrp, op, ch_num=0, sleeptime=0.5):
    """ Step the USRP's oscillator through a list of frequencies """

    freq_list = set_freq_list()
    epoch_start = datetime(1970, 1, 1)
    timestr = '%H:%M:%S.%f'

    # Begin infinite transmission loop
    freq = 0
    while 1:
        gpstime_secs = usrp.get_time_now().get_real_secs()
        gpstime = epoch_start + timedelta(seconds=gpstime_secs)
        gpstime_next = epoch_start + timedelta(seconds=gpstime_secs + 1)
        # Change frequency each time we hit a new time in the list, otherwise hold the existing note
        if ((gpstime_next.second) in freq_list.keys()) and (freq != freq_list[gpstime_next.second]):
            freq = freq_list[gpstime_next.second]

            usrp.set_command_time(
                                  uhd.time_spec(np.ceil(gpstime_secs)),
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
            gpstime_secs = usrp.get_time_now().get_real_secs()
            gpstime = epoch_start + timedelta(seconds=gpstime_secs)
            print('Tuned to %s MHz by GPS time %s' % (str(freq).rjust(4), gpstime.strftime(timestr)))

        time.sleep(sleeptime)


def set_freq_list():
    # time, freq (MHz)
    # This could be set in a text file, but it's important to understand the function above when choosing times
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


