import numpy as np
import time
import scipy
np.random.seed(0)

arr = np.random.rand(524288)

time.sleep(1)

a = np.fft.fft(arr)

time.sleep(1)

a = scipy.fft.fft(arr)
