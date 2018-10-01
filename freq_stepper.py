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
                   out_fname=None,
                   time_source='usrp',
):
    """ Step the USRP's oscillator through a list of frequencies """

    freq_list = set_freq_list()
    if out_fname:
        with open(out_fname, 'a') as f:
            f.write('Tune time (UT)   Freq (MHz)   Tune sample\n')

    # Check for GPS lock
    if time_source == 'usrp':
        while not usrp.get_mboard_sensor("gps_locked", 0).to_bool():
            print("waiting for gps lock...")
            time.sleep(5)
    elif time_source == 'octoclock':
        # Instantiate Octoclock object
        pdb.set_trace()
        clock = uhd.usrp.multi_usrp_clock.make();
"""
        // Instantiate multi_usrp object containing all devices (must supply IP
        addresses)
        multi_usrp::sptr usrp =
        multi_usrp::make("addr0=<IP_addr_0>,addr1=<IP_addr_1>,addr3=<IP_addr_3>,addr4=<IP_addr_4>");
        for (size_t mboard = 0; mboard < usrp->get_numb_mboards; mboard++)
        {
            // Set to external references
            usrp->set_clock_source("external");
            usrp->set_time_source("external");
            // Wait for PLL to lock (there really should be some sort of timeout
        here)
            while (not usrp->get_mboard_sensor("ref_locked", mboard).to_bool())
            {
                boost::this_thread::sleep(boost::posix_time::milliseconds(1);
            }
        }

        // Get time from Octoclock GPSDO
        clock.get_sensor("gps_time");    // alignment with PPS edge is unknown with
        first query
        time_t gps_time = clock.get_sensor("gps_time").to_int();    // second
        successive query will return the time as soon as possible after PPS edge
        for (size_t mboard = 0; mboard < usrp->get_numb_mboards; mboard++)
        {
            usrp->set_time_next_pps(uhd::time_spec_t(gps_time+1), mboard);
        }

"""



       multi_usrp_clock.make() 
        

    # Begin infinite transmission loop
    freq = 0
    while 1:
        gpstime = datetime.utcfromtimestamp(usrp.get_mboard_sensor("gps_time"))
        gpstime_next = gpstime + dt.timedelta(seconds=1)

        # Change frequency each time we hit a new time in the list, otherwise hold the existing note
        if ((gpstime_next.second) in freq_list.keys()) and (freq != freq_list[gpstime_next.second]):
            freq = freq_list[gpstime_next.second]

            # Specify tune time on the first exact sample after listed time
            tune_time_secs = np.ceil(gpstime_secs)

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

            # Optionally write out the shift samples of each frequency
            if out_fname:
                tune_time = drf.util.sample_to_datetime(tune_time_rsamples, op.samplerate)
                tune_sample = int(np.uint64(tune_time_secs * ch_samplerate_ld))
                gps_lock = usrp.get_mboard_sensor("gps_locked").to_bool()
                with open(tune_time.strftime(out_fname), 'a') as f:
                    f.write('GPS lock status: %s' % gps_lock)
                    f.write('%s %s %i\n' % (tune_time.strftime('%Y/%m/%d-%H:%M:%S.%f'), str(freq).rjust(4), tune_sample))
          
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
                print('Tuned to %s MHz by GPS time %s' % (str(freq).rjust(4), gpstime.strftime('%Y%b%d %H:%M:%S.%f')))

        time.sleep(sleeptime)


def set_freq_list():
    # time, freq (MHz)
    # This could be set in a text file, but it's important to understand the function above when choosing times
    return {
             0: 4.5,
             2: 4.6,
             4: 4.7,
             6: 4.8,
             8: 4.9,
            10: 5,
            12: 5.1,
            14: 5.2,
            16: 5.3,
            18: 5.4,
            20: 5.5,
            22: 4.6,
            24: 4.7,
            26: 4.8,
            28: 4.9,
            30: 5,
            32: 5.1,
            34: 5.2,
            36: 5.3,
            38: 5.4,
            40: 5.5,
            42: 4.6,
            44: 4.7,
            46: 4.8,
            48: 4.9,
            50: 5,
            52: 5.1,
            54: 5.2,
            56: 5.3,
            58: 5.4,
           }
    """

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

