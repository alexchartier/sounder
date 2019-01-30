import numpy as np
import numpy.matlib
import pdb
import pickle
import os
import datetime as dt
import glob
import matplotlib.dates as mdates
import matplotlib.colors
from argparse import ArgumentParser


"""
Plot multi-frequency returns as a function of altitude and time
"""


def save_daily_files(op):

    op.chdir = os.path.join(op.datadir, os.path.join('prc_analysis', op.ch))
    print('Processing daily plots in %s' % op.chdir)
    # Set up output dir
    op.outdir = os.path.join('/'.join(op.chdir.split('/')[:-1]), 'daily/data/')
    op.plotdir = os.path.join('/'.join(op.chdir.split('/')[:-1]), 'daily/plots/')
    try:
        os.makedirs(op.outdir)
    except:
        None
    try:
        os.makedirs(op.plotdir)
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
    for root, dn, filenames in os.walk(os.path.join(op.chdir, 'spectra')):
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
            for freq, vals in meta.items():
                data[freq] = {}
                data[freq]['time'] = []
                data[freq]['range'] = vals['range']
                data[freq]['doppler'] = vals['doppler']
        #try:
        for dirn in dn:
            day = dt.datetime.strptime(dirn, '%Y%m%d')
            if day <= startday:
                continue

            daydir = os.path.join(root, dirn)
            for fn in os.listdir(daydir):
                # Get frequencies and times from the filename
                freq = float(fn.split('_')[0]) 
                tod = dt.datetime.strptime(fn.split('_')[2], '%H%M%S')
                data[freq]['time'].append(
                    day + dt.timedelta(
                        hours=tod.hour, 
                        minutes=tod.minute, 
                        seconds=tod.second,
                    )
                )

                # load range, doppler and intensity
                with open(os.path.join(daydir, fn), 'rb') as f:
                    spec = pickle.load(f)
                for k, v in spec.items():
                    try:
                        data[freq][k].append(np.squeeze(v.toarray()))
                    except:
                        data[freq][k] = [np.squeeze(v.toarray()),]

            # Save daily files
            out_fname = os.path.join(op.outdir, day.strftime('%Y%b%d_analysis.pkl')) 
            with open(out_fname, 'wb') as f:
                print('Writing to %s' % out_fname)
                pickle.dump(data, f)

            if not op.noplot:
                plot(out_fname)

            # store what day we got up to
            with open(datpath, 'wb') as f:
                pickle.dump(day, f)
            #except:
            #    None
    return op


def plot(in_fname, plot_fname=None, use_int_pwr=False):
    with open(in_fname, 'rb') as f:
        data = pickle.load(f)
    params = {
        'font.size': 18,
    }
    plt.rcParams.update(params)
     
    for freq, spectra in data.items():
        for k, v in spectra.items():
            spectra[k] = np.array(v)

        plt.style.use('dark_background')
        fig, ax = plt.subplots(figsize=(6, 6))

        # Normalize power
        if use_int_pwr:
            A = spectra['int_pwr']
            A -= 220  # Subtract off noise floor
            A /= 100  # Guessing high power = 100
            A[A < 0] = 0 
            A[A > 1] = 1 
        else:  # use max power
            A = spectra['max_pwr_db']
            # take out the noise floor? Probably ~6 dB
            A -= 6  
            A[A < 0] = 0
            A = 10 ** (A / 10)  # Convert to units of linear power 
            normfac = 15
            A /= normfac  # normalize to X linear power units (at least 10 is good)
            A[A > 1] = 1  # Saturate above normalization

        sortind = np.argsort(spectra['time'])
        # Set plot limits, labels
        rmax = np.array([r.max() for r in spectra['range']])
        ax.grid()
        ax.set_xlabel('Time (UT)')
        ax.set_ylabel('Virtual Range (km)')

        mdat = [mdates.date2num(d) for d in spectra['time'][sortind]]
        myFmt = mdates.DateFormatter('%H:%M')
        ax.xaxis.set_major_formatter(myFmt)
        fig.autofmt_xdate()
      
        # Set colours
        vmin = spectra['doppler'].min()
        vmax = spectra['doppler'].max()
        # vmin = -50
        # vmax = 50
        norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
        sdv = spectra['dop_vel'].copy()[sortind, :]
        cmap = plt.get_cmap('seismic')
        img_array = cmap(norm(sdv))
        img_array[..., 3] = A
        
        # Plot 
        im = ax.imshow(
            img_array, 
            cmap=cmap, vmin=vmin, vmax=vmax, 
            interpolation='none', 
            aspect='auto',
            extent=[mdat[0], mdat[-1], 0, rmax.max()], 
        )

        cbar = plt.colorbar(im)
        cbar.set_label('Doppler velocity (m/s)')
        plt.title('%2.1f MHz, %s' % (freq, spectra['time'][0].strftime('%Y/%b/%d'))) 

        figdir = os.path.join(
            op.plotdir, spectra['time'][0].strftime('rtd_%Y%b%d') + '_%2.2f_MHz.png' % freq,
        )
        plt.savefig(figdir)
        print('Saving to %s'% figdir)


if __name__ == '__main__':
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates

    desc = 'daily plots of range-time-doppler at each frequency in the specified directory'
    parser = ArgumentParser(description=desc)

    parser.add_argument(
        'datadir', help='''Data directory to analyze.''',
    )   
    parser.add_argument(
        '-c', '--ch', default='hfrx',
        help='''Channel name of data to analyze. (default: %(default)s)'''
    )   
    parser.add_argument(
        '-x', '--delete_old', action='store_true', default=False,
        help='''Delete existing processed files.''',
    )
    parser.add_argument(
        '-np', '--noplot', action='store_true', default=False,
        help='''Produce range-doppler plots''',
    )   

    op = parser.parse_args()
    op.datadir = os.path.abspath(op.datadir)

    op = save_daily_files(op)
