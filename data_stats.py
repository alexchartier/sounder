import numpy as np
import pdb 
import scipy.io
import datetime as dt
import h5py
import pickle
import matplotlib.pyplot as plt


def main():
    hf_fname_fmt = 'data/prc_analysis/no_badfrq/daily/data/%Y%b%d_analysis.pkl'
    start_time = dt.datetime(2019, 3, 1)
    end_time = dt.datetime(2019, 3, 14)
    mcm_lat, mcm_lon = -77.8564, 166.6881
    time = start_time

    # Concatenate daily files
    while time < end_time:
        with open(time.strftime(hf_fname_fmt), 'rb') as f:
            dl = pickle.load(f)
        if 'data' in dir():
            for key, val in dl.items():
                for k, v in val.items():
                    if isinstance(v, list):
                        if (k in data[key].keys()):
                            data[key][k].extend(v)
                        else:
                            data[key][k] = v
        else:
            data = dl
        time += dt.timedelta(days=1)

    
    freqs = data.keys()
    freqs.sort()
    good_freqs = []
    for freq in freqs:
        if len(data[freq]['max_pwr_db']) > 100:
            good_freqs.append(freq)

    fig, ax = plt.subplots(len(good_freqs), 4)
    for ind, freq in enumerate(good_freqs[::-1]):
        entry = data[freq]
        rg = entry['range']
        dop = entry['dop_vel']

        ranges = []
        powers = []
        dopplers = []
        for ind2, pwr in enumerate(entry['max_pwr_db']):
            ranges.append(rg[pwr > 0]) 
            powers.append(pwr[pwr > 0])
            dopplers.append(dop[ind2][pwr > 0])
        powers = np.concatenate(powers)
        ranges = np.concatenate(ranges)
        dopplers = np.concatenate(dopplers)
        uts = np.array([t.hour for t in entry['time']])
        lts = uts + mcm_lon / 360 * 24 
        lts[lts >= 24] -= 24

        # Calculate virtual height
        import nvector as nv
        wgs84 = nv.FrameE(name='WGS84')
        mcm = wgs84.GeoPoint(latitude=mcm_lat, longitude=mcm_lon, z=0, degrees=True)
        zsp = wgs84.GeoPoint(latitude=-90, longitude=0, z=0, degrees=True)
        mcm_zsp = np.sqrt(np.sum(mcm.delta_to(zsp).pvector ** 2)) / 1E3 
        R = 6371
        d0 = mcm_zsp / 2 
        h_p = np.sqrt((ranges / 2) ** 2 - (mcm_zsp / 2) ** 2)
        r_d0 = np.sqrt(R ** 2 - d0 ** 2)
        alts = h_p - (R - r_d0)
        h_p = np.sqrt((5994 / 2) ** 2 - (mcm_zsp / 2) ** 2)
        amax = h_p - (R - r_d0)  # maximum range

        print('%1.1f: %i echoes received' % (freq, len(powers)))

        # Plot histograms
        ax[ind, 0].hist(powers, bins=100)
        ax[ind, 1].hist(alts[np.isfinite(alts)], bins=1000)
        ax[ind, 2].hist(dopplers, bins=1000)
        ax[ind, 3].hist(lts, bins=range(0, 25, 3))

        ax[ind, 0].set_xlim(6.5, 15)
        ax[ind, 1].set_xlim(0, 600)
        ax[ind, 2].set_xlim(-200, 200)
        ax[ind, 3].set_xlim(0, 24)

        ax[ind, 0].set_ylabel('%1.1f MHz' % freq)
        for ct in range(4):
            ax[ind, ct].grid()
            if ind < len(good_freqs) - 1:
                ax[ind, ct].set_xticklabels('')

    ax[ind, 0].set_xlabel('Intensity (dB)')
    ax[ind, 1].set_xlabel('Virt. Ht (km)')
    ax[ind, 2].set_xlabel('Doppler vel. (m/s)')
    ax[ind, 3].set_xlabel('Local Time (hr)')
    
    plt.show() 


if __name__ == '__main__':
    main()













