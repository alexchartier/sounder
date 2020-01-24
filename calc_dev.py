import numpy as np
mcm_zsp = 1353
mcm_zsp = 1750
R = 6371
ranges = np.array([1450, 1800])
d0 = mcm_zsp / 2 
h_p = np.sqrt( (ranges / 2) ** 2 - (mcm_zsp / 2) ** 2)
r_d0 = np.sqrt( R ** 2 - d0 ** 2)
alts = h_p - (R - r_d0)
print(ranges)
print(alts)
