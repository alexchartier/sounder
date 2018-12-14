import matplotlib.pyplot as plt
import numpy as np
import pdb
import pickle
import datetime as dt
import matplotlib

"""
Plot multi-frequency returns as a function of altitude and time
"""
time = dt.datetime(2018, 12, 5)
in_fname = time.strftime('/data/chirp_boat/spectra/spectra_%Y%m%d.pkl')
cutoff = 7.0

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

