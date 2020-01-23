import numpy as np
import pdb


comms_flist = [
    4718, 4770, 4730, 
    5726, 5371, 4770,
    6708, 5400, 5371, 
    9032, 7665, 5400, 
    11256, 7995, 7665, 
    13251, 8010, 7995, 
           8245, 8010,
           9006, 8245,
           11553, 9006,
           11570,
]

comms_flist = np.array(comms_flist).astype('float')
min_diff = 250.

nfreqs = 12
test_flist = np.round(np.linspace(2, 15, nfreqs) * 1E2) * 10
test_flist[3] = 5050
test_flist[4] = 6200
test_flist[5] = 7150
test_flist[6] = 8550
test_flist[7] = 9700
test_flist[8] = 10800 
test_flist[9] = 12000 
diffs = {f: np.abs(comms_flist - f).min() for f in test_flist}
print(np.sort(comms_flist / 1E3))
print('\n')
for ind, k  in enumerate(test_flist):
    print("%i: %02.2f MHz within %i kHz of comms freq" \
        % (ind, k/1E3, diffs[k]))
    if diffs[k] < min_diff:
        print('^^^^ WARNING, too close to comm. freq ^^^^^')

print('\nCopy the following into a frequency list text file\n')
print('shift time (seconds): shift frequency (MHz)')

for ind, k  in enumerate(test_flist):
   print('%i: %2.2f' % (60 / nfreqs * ind, k / 1E3)) 


# high-res list
test_flist_2 = np.concatenate(
    [np.arange(2, 10, 0.1), np.arange(10, 14, 0.2)]
) * 1E3

diffs = np.array([np.abs(comms_flist - f).min() for f in test_flist_2])
test_flist_2 = test_flist_2[diffs > min_diff]
print('\nhighres flist')
for ind, k  in enumerate(test_flist_2):
   print('%i: %2.2f' % (ind, k / 1E3)) 
