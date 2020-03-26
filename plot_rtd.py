import numpy as np
import numpy.matlib
import pdb
import pickle
import os
import datetime as dt
import glob
import matplotlib.dates as mdates
import matplotlib.colors
import matplotlib.pyplot as plt
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
    op.plotdir = os.path.join(op.chdir, 'plots/')
    try:
        os.makedirs(op.outdir)
    except:
        None
    try:
        os.makedirs(op.plotdir)
    except:
        None

    # Get metadata first
    metafiles = glob.glob(os.path.join(specdir, 'meta*.pkl'))
    if len(metafiles) > 0:
        meta = {}
        for fname in metafiles:
            freq = float(fname.split('/')[-1].split('_')[1])
            with open(fname, 'rb') as f:
                meta[freq] = pickle.load(f)

    # Clean up input data directory names
    keys = 'time', 'range', 'doppler', 'pwr'
    dirnames = [os.path.join(specdir, dn) for dn in os.listdir(specdir)]
    dirnames = [dn for dn in dirnames if os.path.isdir(dn)]
    dirnames.sort()

    # Go through individual spectra files and process them into daily files
    out_fname_fmt = os.path.join(op.outdir, '%Y%b%d_analysis.pkl')
    if op.reproc:
        preproc_spectra(dirnames, meta, out_fname_fmt)

    # Ignore days outside the limits 
    good_dirn = []
    if op.daylim:
        for dirn in dirnames:
            day = dt.datetime.strptime(dirn.split('/')[-1], '%Y%m%d')
            if (day >= op.startday) and (day <= op.endday):
                good_dirn.append(dirn)

    # Plot the pre-processed output
    if op.daily:
        for dirn in good_dirn:
            dl = load_daily(dirn, out_fname_fmt)   
            plot(dl, op)
    else:
        data = concat_files(good_dirn, out_fname_fmt)   
        plot(data, op)


def concat_files(good_dirn, out_fname_fmt):
    # Concatenate daily files
    for dirn in good_dirn:
        dl = load_daily(dirn, out_fname_fmt)   
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
    return data


def load_daily(dirn, out_fname_fmt):
    nfiles = len(glob.glob(os.path.join(dirn, '*.pkl')))
    print('Loading %s (%i files)' % (dirn, nfiles))
    day = dt.datetime.strptime(dirn.split('/')[-1], '%Y%m%d')
    with open(day.strftime(out_fname_fmt), 'rb') as f:
        dl = pickle.load(f)
    return dl


def plot(data, op, plot_fname=None):
    params = {
        'font.size': 13,
    }
    plt.rcParams.update(params)
    freqs, tlim = get_freqs_tlim(data, fmin=op.fmin, fmax=op.fmax)

    # Set up the subplots
    if not op.white_bg:
        plt.style.use('dark_background')
    fig, ax = plt.subplots(len(freqs), 1, figsize=(12, 8))
    if not isinstance(ax, np.ndarray):
        ax = np.array([ax,])

    # Run through frequencies and plot
    for ind, freq in enumerate(freqs[::-1]):
        print('plotting freq %2.2f MHz' % freq)
        xlabels = True if ind == len(freqs) - 1 else False
        times = data[freq].pop('time')
        im, colorlabel = plt_frq(
            data[freq], fig, ax[ind], tlim, freq=freq, plottype=op.type,
            xlabels=xlabels, vht=op.virt_ht, daylim=op.daylim, white_bg=op.white_bg,
        )

    fig.subplots_adjust(right=0.8)
    cax = fig.add_axes([0.85, 0.2, 0.03, 0.67])
    cbar = plt.colorbar(im, cax=cax)
    cbar.set_label(colorlabel)

    if op.daylim:
        fig.suptitle('%s to %s' % (\
            times[0].strftime('%Y-%m-%d'),
            times[-1].strftime('%Y-%m-%d'),
        ))
    else:
        fig.suptitle(times[0].strftime('%Y-%m-%d '))

    # Plot or save figures
    timestr = times[0].strftime('%Y%b%d')
    if not op.daily:
        timestr += times[-1].strftime('_to_%Y%b%d')
    fig_fname = os.path.join(
        op.plotdir, '%s_%s.png' % (op.type, timestr),
    )
    if len(freqs) > 0:
        if op.show:
            plt.show()
        else:
            if os.path.isfile(fig_fname):
                os.remove(fig_fname)
            plt.savefig(fig_fname)
            print('Saving to %s'% fig_fname)


def plt_frq(spectra, fig, ax, tlim, freq=None, plottype='power', xlabels=False, vht=False, daylim=False, white_bg=True):
    # Add endpoints to all the times, fill in with NaNs for non-recorded times
    ranges = spectra['range']
    times = spectra['time']
    alts = calc_vht(ranges)
    doppler = spectra['doppler']

    if vht:
        y_ax = alts
        ylabel = 'vHt (km)'
        ylim = (0, 800)
        width = 60
    else:
        y_ax = ranges
        ylabel = 'Rg (km)'
        ylim = (op.rmin, op.rmax)
        width = 60

    times = np.array([times, times + dt.timedelta(seconds=width)]).T.flatten()

    for key in ['max_pwr_db', 'dop_vel']:
        val = spectra[key]
        tdim = val.shape[0]
        vals_ext = np.zeros((tdim * 2, val.shape[1])) * np.nan
        vals_ext[np.arange(tdim) * 2, :] = val 
        spectra[key] = vals_ext

    # Normalize power
    A = spectra['max_pwr_db'].copy()
    sortind = np.argsort(times)

    # Set plot labels
    ax.grid(which='both', linestyle='--')
    ylabel = ('%2.2f MHz\n' + ylabel) % freq if freq else ylabel
    ax.set_ylabel(ylabel)
    
    if xlabels:
        fmstr = '%d %b' if daylim else '%H:%M'
        ax.set_xlabel('Time (UT)')
        ax.xaxis.set_major_formatter(mdates.DateFormatter(fmstr))
        fig.autofmt_xdate()
    else:
        ax.set_xlabel('')
        ax.set_xticklabels('')
  
    # Set colours
    if plottype == 'doppler':
        plotvals = spectra['dop_vel'].copy()[sortind, :]
        plotvals[plotvals == 0] *= np.nan
        cmapn = 'seismic_r' if white_bg else 'seismic'
        cmap = plt.get_cmap(cmapn)
        norm = matplotlib.colors.Normalize(vmin=-400, vmax=400)
        colorlabel = 'Doppler velocity (m/s)'
        title = 'Range-Time-Doppler'

    elif plottype == 'power':
        plotvals = A[sortind, :]
        plotvals[plotvals == 0] *= np.nan
        cmapn = 'gist_heat_r' if white_bg else 'gist_heat'
        cmap = plt.get_cmap(cmapn)
        norm = matplotlib.colors.Normalize(vmin=5, vmax=10)
        colorlabel = 'Intensity (dB)'
        title = 'Range-Time-Intensity'

    # Get rid of NaN alt. entries
    finind = np.isfinite(y_ax)
    plotvals = plotvals[:, finind]
    y_ax = y_ax[finind]

    # Plot 
    im = ax.pcolormesh(
        times[sortind], y_ax, plotvals.T, 
        cmap=cmap, norm=norm, shading='flat', 
    )
    ax.set_xlim(tlim[0], tlim[1])
    ax.set_ylim(ylim)
    ax.grid()
    return im, colorlabel


def preproc_spectra(dirnames, meta, out_fname_fmt):
    for dirn in dirnames:
        nfiles = len(glob.glob(os.path.join(dirn, '*.pkl')))
        print('Processing %s (%i files)' % (dirn, nfiles))
        day = dt.datetime.strptime(dirn.split('/')[-1], '%Y%m%d')

        # Set up the data holder here 
        data = {}  
        for freq, vals in meta.items():
            data[freq] = {}
            data[freq]['time'] = []
            data[freq]['range'] = vals['range']
            data[freq]['doppler'] = vals['doppler']            

        for fn in os.listdir(dirn):
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
            with open(os.path.join(dirn, fn), 'rb') as f:
                spec = pickle.load(f)
            for k, v in spec.items():
                try:
                    data[freq][k].append(np.squeeze(v.toarray()))
                except:
                    data[freq][k] = [np.squeeze(v.toarray()),]

        # Save daily files (concatenated spectra)
        out_fname = day.strftime(out_fname_fmt)
        with open(out_fname, 'wb') as f:
            print('Writing to %s' % out_fname)
            pickle.dump(data, f)
    print('\n\n')


def get_freqs_tlim(data, fmin=None, fmax=None):
    # figure out which frequencies have data
    freqs = []
    tmin = dt.datetime(2050, 1, 1)
    tmax = dt.datetime(1, 1, 1)
    for freq, spectra in data.items():
        for k, v in spectra.items():
            spectra[k] = np.array(v)

        min_cts = 90  # require at least this many counts
        if ('int_pwr' in spectra.keys()) and (spectra['int_pwr'].shape[0] > min_cts):
            freqs.append(freq)
            # figure out time limits along the way
            min_t = np.min(spectra['time']) 
            max_t = np.max(spectra['time']) 
            if min_t < tmin:
                tmin = min_t
            if max_t > tmax:
                tmax = max_t
    freqs = np.array(freqs)

    if fmin:
        freqs = freqs[freqs >= np.float(op.fmin)]
    if fmax:
        freqs = freqs[freqs <= np.float(op.fmax)]

    try:
        freqs.sort()
    except:
        None

    if op.tmin:
        tmin = tmin.replace(hour=int(op.tmin), minute=0, second=0)
    else:
        tmin = tmin.replace(hour=0, minute=0, second=0)
    if op.tmax:
        if int(op.tmax) == 24:
            tmax = tmax + dt.timedelta(days=1)
            tmax = tmax.replace(hour=0, minute=0, second=0)
        else:
            tmax = tmax.replace(hour=int(op.tmax), minute=0, second=0)

    else:
        tmax = tmax.replace(hour=23, minute=59, second=59)
    # Abort if no data 
    if freqs.shape[0] == 0:
        print('aborting')
        return

    tlim = tmin, tmax
    return freqs, tlim


def calc_vht(ranges):
    # Calculate distance between McMurdo and Pole
    mcm_zsp, midpt_depth = calc_dist(0, -77.8564, 166.6881, 0, -90, 0)
    alts = np.sqrt((ranges / 2) ** 2 - (mcm_zsp / 2) ** 2) - midpt_depth
    return alts


def calc_dist(alt1, lat1, lon1, alt2, lat2, lon2,):
    import nvector as nv
    wgs84 = nv.FrameE(name='WGS84')
    pt1 = wgs84.GeoPoint(latitude=lat1, longitude=lon1, z=alt1, degrees=True)
    pt2 = wgs84.GeoPoint(latitude=lat2, longitude=lon2, z=alt2, degrees=True)

    # Great circle dist
    # dist_gc = np.sqrt(np.sum(pt1.delta_to(pt2).pvector ** 2)) / 1E3

    # Straight-line dist
    dist_strt = np.sqrt(np.sum((pt1.to_ecef_vector().pvector - pt2.to_ecef_vector().pvector) ** 2)) / 1E3
   
    # midpoint between the two 
    midpt = nv.GeoPath(pt1, pt2).interpolate(0.5).to_geo_point()
    midpt_ll = [midpt.latitude, midpt.longitude]

    # dist. from straight-line midpoint up to ground level
    hypot = np.sqrt(np.sum((pt1.to_ecef_vector().pvector - midpt.to_ecef_vector().pvector) ** 2)) / 1E3
    midpt_depth = np.sqrt(hypot ** 2 - (dist_strt / 2) ** 2)

    return dist_strt, midpt_depth




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
        '-a', '--virt_ht', action='store_true', default=False,
        help='''Plot against virtual height instead of range.''',
    )
    parser.add_argument(
        '-s', '--show', action='store_true', default=False,
        help='''Show the figure instead of saving it''',
    )   
    parser.add_argument(
        '-t', '--type', default='power',
        help='''Choose whether to plot doppler or power on colour axis''',
    )
    parser.add_argument(
        '-wb', '--white_bg', action='store_true', default=False,
        help='''Plot on a white background''',
    )
    parser.add_argument(
        '-x', '--reproc', action='store_true', default=False,
        help='''Repeat processing for all files.''',
    )
    parser.add_argument(
        '-fmin', '--fmin', default=None,
        help='''Min frequency (MHz)''',
    )
    parser.add_argument(
        '-fmax', '--fmax', default=None,
        help='''Max frequency (MHz)''',
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
        '-rmin', '--rmin', default=1300,
        help='''Min plot range (km)''',
    )
    parser.add_argument(
        '-rmax', '--rmax', default=2000,
        help='''max plot range (km)''',
    )
    parser.add_argument(
        '-d', '--daylim', default=None,
        help='''plot days (yyyy,mm,dd, yyyy,mm,dd)''',
    )
    parser.add_argument(
        '-dy', '--daily', action='store_true', default=False,
        help='''make separate daily plots instead of one long one''',
    )
    
    op = parser.parse_args()
    tn = [int(d) for d in op.daylim.split(',')]
    op.startday = dt.datetime(*tn[:3])
    op.endday = dt.datetime(*tn[3:])
    op.datadir = os.path.abspath(op.datadir)
    op.rmin=int(op.rmin)
    op.rmax=int(op.rmax)
    save_daily_files(op)
