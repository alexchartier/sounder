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


def main():
    # Test out the frequency stepper
    freq_list_fname = 'freq_list_default.txt'
    step([], [], freq_list_fname=freq_list_fname)


def step(usrp, op, 
    ch_num=0, 
    sleeptime=0.1, 
    freq_list_fname=None,
    flog_fname=None,
    lock_fname=None,
    time_source='GPS',
    timestr='%Y/%b/%d %H:%M:%S',
):
    """ Step the USRP's oscillator through a list of frequencies """
    if freq_list_fname:
        freq_list = get_freq_list(freq_list_fname) if freq_list_fname else set_freq_list()
    else:
        freq_list = set_freq_list()

    print('Starting freq_stepper')

    prev_lock = False
    # Check for GPS lock
    while not usrp.get_mboard_sensor("gps_locked", 0).to_bool():
        print("waiting for gps lock...")
        time.sleep(5)
    assert usrp.get_mboard_sensor("gps_locked", 0).to_bool(), "GPS still not locked"

    # Begin infinite transmission loop
    freq = 0
    while 1:

        # Set USRP time (necessary to know what sample number we shifted frequencies at)
        usrptime_secs = usrp.get_time_now().get_real_secs()  

        if time_source == 'GPS':
            # Set GPS time (necessary to sync operations between the transmitter and receiver)
            gpstime = datetime.utcfromtimestamp(usrp.get_mboard_sensor("gps_time"))
            time_next = pytz.utc.localize(gpstime) + timedelta(seconds=1)
        elif time_source == 'USRP':
            time_next = drf.util.epoch + timedelta(seconds=usrptime_secs + 1)

        # Calculate the samplerate
        try:  
            ch_samplerate_frac = op.ch_samplerates_frac[ch_num]
            ch_samplerate_ld = (
                np.longdouble(ch_samplerate_frac.numerator)
                / np.longdouble(ch_samplerate_frac.denominator)
            )
        except:
            ch_samplerate_ld = op.samplerate
            
        # Frequency shifting block
        #    Change frequency each time we hit a new time in the list, otherwise hold the existing note
        if ((time_next.second) in freq_list.keys()) and (freq != freq_list[time_next.second]):
            tune_time = time_next
            freq = freq_list[time_next.second]

            # Specify USRP tune time on the first exact sample after listed time
            # tune_time_secs = (tune_time - drf.util.epoch).total_seconds()
            tune_time_secs = usrp.get_time_last_pps().get_real_secs() + 1
            tune_time_rsamples = np.ceil(tune_time_secs * op.samplerate)
            tune_time_secs = tune_time_rsamples / op.samplerate


            gps_lock = usrp.get_mboard_sensor("gps_locked").to_bool()
            print('GPS lock status: %s' % gps_lock)
            timestr = tune_time.strftime('%Y/%m/%d-%H:%M:%S')
            if lock_fname:
                if gps_lock != prev_lock:
                    with open(tune_time.strftime(lock_fname), 'a+') as f:
                        f.write('GPS lock status: %s at %s' % (gps_lock, timestr))
                prev_lock = gps_lock

            # Optionally write out the shift samples of each frequency
            tune_sample = int(np.uint64(tune_time_secs * ch_samplerate_ld))
            if flog_fname:
                # Change to 'a' to append
                with open(tune_time.strftime(flog_fname), 'w') as f:
                    f.write('%s %s %i\n' % (timestr, str(freq).rjust(4), tune_sample))
          
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
            if op.verbose:
                print('Tuned to %s MHz at %s (sample %i)' % \
                        (str(freq).rjust(4),
                        tune_time.strftime(timestr), 
                        tune_sample, 
                        )
                )
                """
                gpstime = datetime.utcfromtimestamp(usrp.get_mboard_sensor("gps_time"))
                usrptime = drf.util.epoch + timedelta(seconds=usrp.get_time_now().get_real_secs())
                print('GPS tune time:  %s\nUSRP tune time: %s' %
                    (gpstime.strftime(timestr),
                    usrptime.strftime(timestr))
                )
                """
        time.sleep(sleeptime)


def set_dev_time(usrp):
    # 7)  Verify that usrp->get_time_last_pps() and usrp->get_mboard_sensor("gps_time") return the same time.
    
    # while usrp.get_time_last_pps().get_real_secs() + 1 != usrp.get_mboard_sensor("gps_time").to_real():
    while usrp.get_time_last_pps().get_real_secs() != usrp.get_mboard_sensor("gps_time").to_real():
        print(usrp.get_time_last_pps().get_real_secs())
        print(usrp.get_mboard_sensor("gps_time").to_real())
        # 1)  Poll on usrp->get_mboard_sensor("gps_locked") until it returns true
        while not usrp.get_mboard_sensor("gps_locked", 0).to_bool():
            print("Waiting for gps lock...")
            time.sleep(5)
        print("...GPS locked!")

        # 2)  Poll on usrp->get_time_last_pps() until a change is seen.
        pps = usrp.get_time_last_pps()
        while usrp.get_time_last_pps() == pps:
            time.sleep(0.1)

        # 3)  Sleep 200ms (allow NMEA string to propagate)
        time.sleep(0.2)

        # 4)  Use "usrp->set_time_next_pps(uhd::time_spec_t(usrp->get_mboard_sensor("gps_time").to_int()+1));" to set the time
        usrp.set_time_next_pps(uhd.time_spec_t(usrp.get_mboard_sensor("gps_time").to_int() + 2))

        # 5)  Poll on usrp->get_time_last_pps() until a change is seen.
        pps = usrp.get_time_last_pps()
        while usrp.get_time_last_pps() == pps:
            time.sleep(0.1)

        # 6)  Sleep 200ms (allow NMEA string to propagate)
        time.sleep(0.2)
        print('USRP last PPS = %i, GPSDO = %i' % (\
            usrp.get_time_last_pps().get_real_secs(), 
            usrp.get_mboard_sensor("gps_time").to_real()
        ))

    print('time set')

def get_freq_list(freq_list_fname):
    freq_list = {}
    with open(freq_list_fname, 'r') as f:
        for line in f:
            try:
                k, v = line.split(':')
                freq_list[int(k)] = float(v) 
            except:
                None
        assert len(freq_list) > 0, "Could not load %s" % freq_list_fname
    return freq_list


def set_freq_list():
    # shift time (seconds), freq (MHz)
    return {
             0: 3.0,
            10: 4.0,
            20: 5.1,
            30: 8.0,
            40: 12.0,
            50: 16.0,
           }

if __name__ == '__main__':
    main()
