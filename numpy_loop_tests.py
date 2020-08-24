import numpy as np
import time
np.random.seed(0)

print("build_array")
arr = np.random.rand(1000,100,100)

time.sleep(1)

print("start_run")
for _ in range(100):
    a = np.mean(arr)
    time.sleep(1)
