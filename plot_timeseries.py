import matplotlib.pyplot as plt
import numpy as np
import numpy.matlib
import pdb
import pickle
import os
import datetime as dt
import glob
import pandas
import matplotlib
from argparse import ArgumentParser


"""
Plot multi-frequency returns as a function of altitude and time
"""
def main(op):

    op = save_daily_files(op)


def save_daily_files(op):
    # TODO: Add last.dat and delete_old functionality similar to analyze_prc
    # TODO: Store directory info in op structure


    # Set up output dir
    op.outdir = os.path.join('/'.join(op.indir.split('/')[:-1]), 'daily')
    try:
        os.makedirs(op.outdir)
    except:
        None

    # Remove old files if necessary
    datpath = os.path.join(op.outdir, 'lastplot.pkl')
    if op.delete_old:
        for root, dirnames, filenames in os.walk(op.outdir):
            for f in filenames:
                if f.endswith(('.png', '.pkl')):
                    os.remove(os.path.join(root, f)) 

    # See where we got up to before.
    if os.path.isfile(datpath):
        try:
            with open(datpath, 'rb') as f:
                startday = pickle.load(f)
        except:
            startday = dt.datetime(1900, 1, 1)
    else: 
        startday = dt.datetime(1900, 1, 1)

    keys = 'time', 'range', 'doppler', 'pwr'
    for root, dn, filenames in os.walk(op.indir):
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
            if day < startday:
                continue

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
            out_fname = os.path.join(op.outdir, day.strftime('%Y%b%d_analysis.pkl')) 
            with open(out_fname, 'wb') as f:
                print('Writing to %s' % out_fname)
                pickle.dump(data, f)

            if op.plot:
                plot(out_fname)

            # store what day we got up to
            with open(datpath, 'wb') as f:
                pickle.dump(day, f)
        except:
            None
    return op


def plot(in_fname, plot_fname=None):
    with open(in_fname, 'rb') as f:
        data = pickle.load(f)
    params = {
        "ytick.color" : "w",
        "xtick.color" : "w",
        "axes.labelcolor" : "w",
        "axes.edgecolor" : "w",
        'font.size': 16,
    }
    plt.rcParams.update(params)
     
    for freq, spectra in data.items():
        spectra['time'] = np.array(spectra['time'])
        # Set plot limits, labels
        fig, ax = plt.subplots(figsize=(10,10))
        ax.set_xlim([spectra['time'].min(), spectra['time'].max()])
        rmax = np.array([r.max() for r in spectra['range']])
        ax.set_ylim([0, rmax.max()])
        ax.grid()
        ax.set_xlabel('Time (UT)')
        ax.set_ylabel('Virtual Height (km)')

        # Set colours
        ax.set_facecolor('black')
        fig.patch.set_facecolor('black')
        normalize = matplotlib.colors.Normalize(vmin=-500, vmax=500)
        cmap = matplotlib.cm.get_cmap('seismic')

        # Plot each instant in time
        for t in spectra['time']:
            timeind = np.where(spectra['time'] == t)[0][0]
            ranges = spectra['range'][timeind]
            doppler = spectra['doppler'][timeind]
            colors = [cmap(normalize(value)) for value in doppler]
            tr = np.matlib.repmat(t, 1, ranges.shape[0])
            cax = ax.scatter(tr, ranges, s=5, c=doppler, cmap=cmap, norm=normalize)

        cbar = fig.colorbar(cax)
        cbar.set_label('Doppler velocity (m/s)')
        plt.title('%2.2f MHz, %s' % (freq, t.strftime('%Y %b %d')), color='w')

        plt.show()

if __name__ == '__main__':
    desc = 'daily plots of range-time-doppler at each frequency in the specified directory'
    parser = ArgumentParser(description=desc)

    parser.add_argument(
        'indir', help='''Data directory to analyze.''',
    )   
    parser.add_argument(
        '-x', '--delete_old', action='store_true', default=False,
        help='''Delete existing processed files.''',
    )
    parser.add_argument(
        '-p', '--plot', action='store_true', default=False,
        help='''Produce range-doppler plots''',
    )   

    op = parser.parse_args()
    op.indir = os.path.abspath(op.indir)

    main(op)
