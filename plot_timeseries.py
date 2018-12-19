import matplotlib.pyplot as plt
import numpy as np
import pdb
import pickle
import os
import datetime as dt
import glob
import pandas
import matplotlib

"""
Plot multi-frequency returns as a function of altitude and time
"""
def main():

    indir = '/data/chirp_notx/prc_analysis/hfrx/spectra/'
    save_daily_files(indir)
    plot()


def save_daily_files(indir):
    keys = 'time', 'range', 'doppler', 'pwr'
    for root, dn, filenames in os.walk(indir):
        # Get metadata first
        metafiles = glob.glob(os.path.join(root, 'meta*.pkl'))
        if len(metafiles) > 0:
            meta = {}
            for fname in metafiles:
                freq = float(fname.split('/')[-1].split('_')[1])
                with open(fname, 'rb') as f:
                    meta[freq] = pickle.load(f)

            # Set up the data holder here 
            data = {}  
            for freq in meta.keys():
                data[freq] = {}
                for key in keys:
                    data[freq][key] = []
                
        try:
            day = dt.datetime.strptime(root.split('/')[-1], '%Y%m%d')
            for fn in filenames:
                # Get frequencies and times from the filename
                freq = float(fn.split('_')[0]) 
                tod = dt.datetime.strptime(fn.split('_')[2], '%H%M%S')
                data[freq]['time'].append(
                    day + dt.timedelta(hours=tod.hour, 
                    minutes=tod.minute, seconds=tod.second)
                )
                # load range, doppler and intensity
                with open(os.path.join(root, fn), 'rb') as f:
                    spec = pickle.load(f)
                dopind, rgind = spec.nonzero()
                data[freq]['doppler'].append(meta[freq]['Doppler (m/s)'][dopind])
                data[freq]['range'].append(meta[freq]['Range (km)'][rgind])
                '''
                data[freq]['doppler'].append(meta[freq]['doppler'][dopind])
                data[freq]['range'].append(meta[freq]['range'][rgind])
                pdb.set_trace()
                '''
                sparr = spec.toarray()
                data[freq]['pwr'].append(np.array(sparr[sparr > 0]))

            # Save daily files
            outdir = os.path.join('/'.join(root.split('/')[:-2]), 'daily')
            try:
                os.makedirs(outdir)
            except:
                None
            out_fname = os.path.join(outdir, day.strftime('%Y%b%d_analysis.pkl')) 
            with open(out_fname, 'wb') as f:
                print('Writing to %s' % out_fname)
                pickle.dump(data, f)
        except:
            None

with open(in_fname, 'rb') as f:
    spectra = pickle.load(f)

for k, v in spectra.items():
    try:
        spectra[k] = np.array(v)
    except:
        spectra[k] = v[0]

low_alts = []
mean_vels = []
times = []

for t in spectra['time']:
    timeind = spectra['time'] == t
    M = spectra['M'][timeind].squeeze()
    alts = spectra['range'][timeind].squeeze() / 2 / 1E3
    vels = spectra['doppler'][timeind].squeeze()  * -1
    inds = np.where(M > cutoff)
    
    if np.any(M > cutoff):
        low_alt_ind = inds[1].min()
        low_alts.append(alts[low_alt_ind])
        mean_vels.append(np.mean(vels[inds[0][inds[1] == low_alt_ind]]))
        times.append(t)

cmap = matplotlib.cm.get_cmap('seismic')
normalize = matplotlib.colors.Normalize(vmin=-50, vmax=50)
colors = [cmap(normalize(value)) for value in mean_vels]
fig, ax = plt.subplots(figsize=(10,10))
ax.set_xlim([times[0], times[-1]])
ax.set_ylim([0, 400])
ax.set_facecolor('black')
ax.grid()
ax.set_xlabel('Time (UT)')
ax.set_ylabel('Virtual Height (km)')
ax.scatter(times, low_alts, color=colors)
cax, _ = matplotlib.colorbar.make_axes(ax)
cbar = matplotlib.colorbar.ColorbarBase(cax, cmap=cmap, norm=normalize)
plt.show()

