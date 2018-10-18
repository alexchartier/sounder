
"""
Test the operation of the octoclock within a Python script
"""

import pdb 
from datetime import datetime, timedelta
import numpy as np  
import time 
from gnuradio import uhd 
import pytz
import digital_rf as drf 

pdb.set_trace()
octoclock = uhd.usrp.multi_usrp_clock.make();

