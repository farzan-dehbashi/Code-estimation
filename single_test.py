'''
To run:
numpy -> python3 single_test.py numpy

scipy -> python3 single_test.py scipy

'''

import numpy as np
import scipy
import sys

if sys.argv[-1] == 'numpy':
    print(np.fft.test(verbose=3)
else:
    import scipy.stats
    print(scipy.stats.test(verbose=3))
