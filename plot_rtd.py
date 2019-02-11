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
    assert os.path.isdir(op.chdir), 'Directory not found: %s' % op.chdir
    specdir = os.path.join(op.chdir, 'spectra')
    assert os.path.isdir(specdir), 'No spectra found in chdir - check %s' % op.chdir
    print('Processing daily plots in %s' % op.chdir)
    # Set up output dir
    op.outdir = os.path.join(op.chdir, 'daily/data/')
    op.plotdir = os.path.join(op.chdir, 'daily/plots/')
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

    # See where we got up to before.
    if os.path.isfile(datpath) and not op.restart:
            try:
                with open(datpath, 'rb') as f:
                    startday = pickle.load(f)
            except:
                startday = dt.datetime(1900, 1, 1)
    else: 
        startday = dt.datetime(1900, 1, 1)

    keys = 'time', 'range', 'doppler', 'pwr'
    for root, dn, filenames in os.walk(specdir):
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


            # Save daily files (concatenated spectra)
            out_fname = os.path.join(op.outdir, day.strftime('%Y%b%d_analysis.pkl')) 
            with open(out_fname, 'wb') as f:
                print('Writing to %s' % out_fname)
                pickle.dump(data, f)

            # Plot the pre-processed output
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
        'font.size': 15,
    }
    plt.rcParams.update(params)

    # First, figure out which frequencies have data
    freqs = []
    tmin = dt.datetime(2050, 1, 1)
    tmax = dt.datetime(1, 1, 1)

    for freq, spectra in data.items():
        for k, v in spectra.items():
            spectra[k] = np.array(v)

        if ('int_pwr' in spectra.keys()) and (spectra['int_pwr'].shape[0] > 10):
            freqs.append(freq)
            # figure out time limits along the way
            min_t = np.min(spectra['time']) 
            max_t = np.max(spectra['time']) 
            if min_t < tmin:
                tmin = min_t
            if max_t > tmax:
                tmax = max_t

    freqs = np.array(freqs)
    if op.fmin:
        freqs = freqs[freqs > np.float(op.fmin)]
    if op.fmax:
        freqs = freqs[freqs < np.float(op.fmax)]

    try:
        freqs.sort()
    except:
        None

    if op.tmin:
        tmin = tmin.replace(hour=int(op.tmin), minute=0, second=0)
    if op.tmax:
        tmax = tmax.replace(hour=int(op.tmax), minute=0, second=0)

    # Abort if no data 
    if freqs.shape[0] == 0:
        print('No spectra found in %s' % in_fname)
        exit()

    # Then plot
    plt.style.use('dark_background')
    fig, ax = plt.subplots(len(freqs), 1, figsize=(12, 8))
    ax = np.array([ax,])
    for ind, freq in enumerate(freqs[::-1]):
        print('plotting freq %2.2f MHz' % freq)

        spectra = data[freq]

        # Add endpoints to all the times, fill in with NaNs for non-recorded times
        ranges = spectra.pop('range')
        times = spectra.pop('time')
        doppler = spectra.pop('doppler')
        times = np.array([times, times + dt.timedelta(seconds=5)]).T.flatten()
    
        if op.geophys:
            import nvector as nv
            wgs84 = nv.FrameE(name='WGS84')
            R = 6371
            mcm = wgs84.GeoPoint(latitude=-77.51, longitude=166.40, z=0, degrees=True)
            zsp = wgs84.GeoPoint(latitude=-90, longitude=0, z=0, degrees=True)
            mcm_zsp = np.sqrt(np.sum(mcm.delta_to(zsp).pvector ** 2)) / 1E3
            d0 = mcm_zsp / 2
            h_p = np.sqrt( (ranges / 2) ** 2 - (mcm_zsp / 2) ** 2)
            r_d0 = np.sqrt( R ** 2 - d0 ** 2)
            alts = h_p - (R - r_d0)
            y_ax = alts
            ylabel = 'Virt. Ht (km)'
            ylim = (100, 800)
        else:
            y_ax = ranges
            ylabel = 'Virt. Rg. (km)'
            ylim = (1400, 1700)

        for key, val in spectra.items():
            tdim = val.shape[0]
            vals_ext = np.zeros((tdim * 2, val.shape[1])) * np.nan
            vals_ext[np.arange(tdim) * 2, :] = val 
            spectra[key] = vals_ext
    
        # Normalize power
        if use_int_pwr:
            A = spectra['int_pwr'].copy()
            A -= 220  # Subtract off noise floor
            A /= 100  # Guessing high power = 100
            A[A < 0] = 0 
            A[A > 1] = 1 
        else:  # use max power
            A = spectra['max_pwr_db'].copy()
            """
            A = 10 ** (A / 10)  # Convert to units of linear power 
            normfac = 10
            A /= normfac  # normalize to X linear power units (10+ is good)
            A[A > 1] = 1  # Saturate above normalization
            """

        sortind = np.argsort(times)

        # Set plot labels
        ax[ind].grid(which='both', linestyle='--')
        ax[ind].set_ylabel(('%2.2f MHz\n' + ylabel) % freq)

        if ind == len(freqs) - 1:
            ax[ind].set_xlabel('Time (UT)')
            ax[ind].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
            fig.autofmt_xdate()
        else:
            ax[ind].set_xlabel('')
            ax[ind].set_xticklabels('')
      
        # Set colours
        if op.type == 'doppler':
            plotvals = spectra['dop_vel'].copy()[sortind, :]
            plotvals[plotvals == 0] *= np.nan
            cmap = plt.get_cmap('seismic')
            norm = matplotlib.colors.Normalize(vmin=-500, vmax=500)
            colorlabel = 'Doppler velocity (m/s)'
            title = 'Range-Time-Doppler'

        elif op.type == 'power':
            plotvals = A[sortind, :]
            plotvals[plotvals == 0] *= np.nan
            cmap = plt.get_cmap('gist_heat')
            norm = matplotlib.colors.Normalize(vmin=5, vmax=10)
            colorlabel = 'Intensity (dB)'
            title = 'Range-Time-Intensity'

        # Get rid of NaN alt. entries
        finind = np.isfinite(y_ax)
        plotvals = plotvals[:, finind]
        y_ax = y_ax[finind]

        # Plot 
        im = ax[ind].pcolormesh(
            times[sortind], y_ax, plotvals.T, 
            cmap=cmap, norm=norm, shading='flat', 
        )
        ax[ind].set_xlim(tmin, tmax)
        ax[ind].set_ylim(ylim)
        ax[ind].grid()

    fig.subplots_adjust(right=0.8)
    cax = fig.add_axes([0.85, 0.2, 0.03, 0.67])
    cbar = plt.colorbar(im, cax=cax)
    cbar.set_label(colorlabel)

    fig.suptitle(times[0].strftime('%Y-%m-%d ') + title)

    figdir = os.path.join(
        op.plotdir, times[0].strftime('rtd_%Y%b%d.png'),
    )
    plt.show()
    plt.savefig(figdir)
    print('Saving to %s'% figdir)


if __name__ == '__main__':
    import matplotlib
    # matplotlib.use('Agg')
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
        '-x', '--restart', action='store_true', default=False,
        help='''Repeat processing for all files.''',
    )
    parser.add_argument(
        '-g', '--geophys', action='store_true', default=True,
        help='''Use derived geophysical parameters instead of raw quantities (e.g. alt instead of range).''',
    )
    parser.add_argument(
        '-np', '--noplot', action='store_true', default=False,
        help='''Produce range-doppler plots''',
    )   
    parser.add_argument(
        '-t', '--type', default='power',
        help='''Choose whether to plot doppler or power on colour axis''',
    )
    parser.add_argument(
        '-tmin', '--tmin', default=None,
        help='''Min plot time (hour)''',
    )
    parser.add_argument(
        '-tmax', '--tmax', default=None,
        help='''max plot time (hour)''',
    )
    parser.add_argument(
        '-fmin', '--fmin', default=None,
        help='''Min frequency (MHz)''',
    )
    parser.add_argument(
        '-fmax', '--fmax', default=None,
        help='''Max frequency (MHz)''',
    )

    op = parser.parse_args()
    op.datadir = os.path.abspath(op.datadir)

    op = save_daily_files(op)
