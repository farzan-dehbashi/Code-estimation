import numpy as np
import time
np.random.seed(0)

time.sleep(1)

print("build_array")
arr = np.random.rand(1000,100,100)

time.sleep(1)

a = arr.flatten()

time.sleep(1)

a = arr.reshape(((self.arr.shape[0]*self.arr.shape[1]*self.arr.shape[2]),))
