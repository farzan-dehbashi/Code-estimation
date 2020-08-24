import numpy as np
import time
np.random.seed(0)

print("making_array")
arr = np.random.rand(1000,100,100)

time.sleep(1)

print("base_function")
a = np.mean(arr)


# -------

time.sleep(1)
a = np.mean(arr)

time.sleep(1)
a = np.median(arr)

time.sleep(1)
a = np.amax(arr)

time.sleep(1)
a = np.amin(arr)

time.sleep(1)
a = arr.reshape(((arr.shape[0]*arr.shape[1]*arr.shape[2]),))

time.sleep(1)
a = arr.flatten()
