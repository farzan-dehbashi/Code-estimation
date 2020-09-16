import numpy as np
import time
np.random.seed(0)


time.sleep(1)

for i in range(10):
    arr = np.random.rand(1000,(i*20),(i*20))
    time.sleep(1)
    a = np.mean(arr)
    time.sleep(1)
