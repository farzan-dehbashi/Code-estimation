import numpy as np
import time
import random
from random import sample
import sys
np.random.seed(0)



class runNumpy(object):
    def __init__(self, arr):
        super(runNumpy, self).__init__()

        self.arr = arr

        # set functions here
        self.funcs = [self.np_mean,self.np_median,self.np_max,self.np_min,self.np_reshape,self.np_flat]

    def callSoft(self):
        funcs_to_run = sample(self.funcs,len(self.funcs))

        print("looping")
        time.sleep(3)
        for i in funcs_to_run:
            time.sleep(1)
            i()


    def np_mean(self):
        print("mean")
        a = np.mean(self.arr)

    def np_median(self):
        print("median")
        a = np.median(self.arr)

    def np_max(self):
        print("max")
        a = np.amax(self.arr)

    def np_min(self):
        print("min")
        a = np.amin(self.arr)

    def np_reshape(self):
        print("reshape")
        a = self.arr.reshape(((self.arr.shape[0]*self.arr.shape[1]*self.arr.shape[2]),))

    def np_flat(self):
        print("flatten")
        a = self.arr.flatten()


print("making_array")
arr = np.random.rand(1000,100,100)

time.sleep(1)

print("base_function_np_mean")
a = np.mean(arr)

# ------- testing

time.sleep(1)

numpy_class = runNumpy(arr)

time.sleep(1)

numpy_class.callSoft()
