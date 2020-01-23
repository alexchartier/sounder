"""
Remove files not in freq_list
"""


import fnmatch
import os
import pdb
from freq_stepper import get_freq_list

flist_fname = 'freq_lists/mcm_new.flist'
dirn = 'data/prc_analysis/no_badfrq/'

flist = get_freq_list(flist_fname)
matches = []
for root, dirnames, filenames in os.walk(dirn):
    for filename in fnmatch.filter(filenames, '*.pkl'):
        matches.append(os.path.join(root, filename))

rm_files = []
for fname in matches:
    try:
        fn_frq = float(fname.split('/')[-1].split('_')[0])
    except:
        fn_frq = float(fname.split('/')[-1].split('_')[1])
    if fn_frq in flist.values():
        print('%s in list' % fname)
    else:
        print('%s NOT in list' % fname)
        rm_files.append(fname)
for fname in rm_files:
    os.remove(fname)
