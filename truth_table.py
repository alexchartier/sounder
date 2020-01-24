import numpy as np
import pdb 
import scipy.io
import datetime as dt
import h5py
import pickle


def main():
    tec_fname = 'data/midas_tec.mat'
    hf_fname_fmt = 'data/prc_analysis/no_badfrq/daily/data/%Y%b%d_analysis.pkl'
    start_time = dt.datetime(2019, 3, 1)
    end_time = dt.datetime(2019, 3, 13)

    # Set up storage
    bins = []
    time = start_time
    while time < end_time:
        bins.append(time)
        time += dt.timedelta(minutes=15) 
    bins = np.array(bins)
    tec_cts = np.zeros(bins.shape)
    hf_cts = np.zeros(bins.shape)

    # Load MIDAS
    Midas = {}
    with h5py.File(tec_fname, 'r') as f:
        for k, v in f.items():
            Midas[k] = v[...]
    Midas['dt'] = []
    for dn in Midas['Time']:
        Midas['dt'].append(datenum_to_datetime(dn))
    Midas['dt'] = np.array(Midas['dt'])
    tind = np.logical_and(Midas['dt'] >= start_time, Midas['dt'] <= end_time)
    for k, v in Midas.items():
        Midas[k] = v[tind]

    TEC_cutoff = 3.0
    for ind, t in enumerate(Midas['dt']):
        if Midas['TEC'][ind] > TEC_cutoff: 
            delta_t = [td.total_seconds() for td in (bins - t)]
            tecind = np.argmin(np.abs(delta_t))
            tec_cts[tecind] = 1
            assert delta_t[tecind] < 1, 'Should be within 1 sec of a bin'

    time = start_time
    while time < end_time:
        # HF counts
        with open(time.strftime(hf_fname_fmt), 'rb') as f:
            HF = pickle.load(f)
        for t in HF[7.2]['time']:
            delta_t = [td.total_seconds() for td in (bins - t)]
            hfind = np.argmin(np.abs(delta_t))
            assert delta_t[hfind] < 15 * 60, 'Should be within 15 minutes of a bin'
            hf_cts[hfind] = 1

        time += dt.timedelta(days=1)

    notec_cts = np.logical_not(tec_cts)
    nohf_cts = np.logical_not(hf_cts)
    TEC_HF = np.float(np.sum(np.logical_and(tec_cts, hf_cts)))
    TEC_noHF = np.float(np.sum(np.logical_and(tec_cts, nohf_cts)))
    noTEC_HF = np.float(np.sum(np.logical_and(notec_cts, hf_cts)))
    noTEC_noHF = np.sum(np.logical_and(notec_cts, nohf_cts)) 
    print('         TEC > %1.1f,   TEC < %1.1f' % (TEC_cutoff, TEC_cutoff))        
    print('7.2 MHz    %i          %i'  % (TEC_HF, noTEC_HF)) 
    print('No 7.2     %i          %i'  % (TEC_noHF, noTEC_noHF)) 
    print('True positive prediction: %2.0f pct' % (100. * TEC_HF / (TEC_HF + TEC_noHF)))
    print('True negative prediction: %2.0f pct' % (100. * noTEC_noHF / (noTEC_noHF + noTEC_HF)))


def datenum_to_datetime(datenum):
    """
    Convert Matlab datenum into Python datetime.

    :param datenum: Date in datenum format
    :return:        Datetime object corresponding to datenum.
    """
    days = datenum % 1
    hours = days % 1 * 24
    minutes = hours % 1 * 60
    seconds = minutes % 1 * 60
    return dt.datetime.fromordinal(int(datenum)) \
           + dt.timedelta(days=int(days)) \
           + dt.timedelta(hours=int(hours)) \
           + dt.timedelta(minutes=int(minutes)) \
           + dt.timedelta(seconds=round(seconds)) \
           - dt.timedelta(days=366)

if __name__ == '__main__':
    main()
