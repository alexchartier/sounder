import pdb
import time
from freq_stepper import get_freq_list
import os
import numpy as np
import shutil
from datetime import datetime

def sort_freqs(datadir, freq_list_fname, fdir='_freqs'):
    # Get frequency information from freq_list
    flist = get_freq_list(freq_list_fname)
    freqs = flist.values()
    shift_secs = flist.keys()
    shift_secs = np.array(shift_secs)
    shift_secs.sort()

    # Make frequency sub-directories
    freq_subdirs = {}
    for freq in freqs:
        f_subdir = os.path.join(os.path.join(datadir, fdir), str(freq))
        freq_subdirs[freq] = f_subdir 
        try: 
            os.makedirs(f_subdir)
        except:
            None
    shutil.copy2(freq_list_fname, f_subdir) 

    # Move the files into frequency-specific subdirs
    subdirs = next(os.walk(datadir))[1]
    for subdir in subdirs:
        if (subdir != 'metadata') and (subdir != fdir):
            subdirn = os.path.join(datadir, subdir)
            files = os.listdir(subdirn)
            time.sleep(1)  # wait a few seconds so as not to move open files
            for f in files:
                try:
                    ts = datetime.utcfromtimestamp(float(f.split('.')[0][3:]))
                    diff = ts.second - shift_secs
                    shift_t = shift_secs[diff == diff[diff >= 0].min()]
                    assert len(shift_t) == 1, 'length is wrong' 
                    shift_freq = flist[shift_t[0]]     
                    shutil.move(
                        os.path.join(subdirn, f), \
                        os.path.join(freq_subdirs[shift_freq], subdirn.split('/')[-1]),
                    )
                except:
                    None

    # Copy over the metadata files
    for freq_subdir in freq_subdirs.values():
        shutil.copy(os.path.join(datadir, 'drf_properties.h5'), freq_subdir)
        new_metadatadir = os.path.join(freq_subdir, 'metadata')
        if os.path.exists(new_metadatadir):
            shutil.rmtree(new_metadatadir)
        shutil.copytree(os.path.join(datadir, 'metadata'), new_metadatadir)

    # Remove empty directories
    for subdir in subdirs:
        subdirn = os.path.join(datadir, subdir)
        try:
            os.rmdir(subdirn)
        except:
            None

if __name__ == '__main__':
    desc = """
        Script for moving odin output into frequency-specific directories
        ahead of processing

    """

    parser = ArgumentParser(description=desc)

    parser.add_argument(
        'datadir', default='/data/spread_spec/',
        help='''Data directory to analyze. (default: %(default)s)''',
    )   
    parser.add_argument(
        '-c', '--ch', default='hfrx',
        help='''Channel name of data to analyze. (default: %(default)s)'''
    )   
    parser.add_argument(
        '-f', '--freq_list', default='freq_lists/freq_list.txt',
        help='''Frequency shift list. (default: %(default)s)'''
    )   

    op = parser.parse_args()

    op.datadir = os.path.abspath(op.datadir)
    sort_freqs(os.path.join(op.datadir, op.ch), op.freq_list)





