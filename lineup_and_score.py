import numpy as np
import sys
from os import listdir
from matplotlib import pyplot as plt

og_path = "power_cunsumption/2k_per_second_"


def getTestingSuite(lang):
    try_paths = [og_path + lang +"/"+f+"/" + lang +"_timestamped_"+f+".npy" for f in listdir(og_path + lang +"/") if f[:3] == "try"]
    traces = []
    for path in try_paths:
        arr = np.load(path)
        mn = []
        for i in range(1,arr.shape[0]):
            if i < 500:
                mn.append(np.mean(arr[:i]))
            else:
                mn.append(np.mean(arr[(i-500):i]))
        traces.append(np.array(mn))
        plt.plot(np.arange(np.array(mn).shape[0]), np.array(mn))
        plt.show()

    traces = np.array(traces) # FIXME: I dont do anything with this yet, just take the first one
    return traces[0]

def getTrace(lang):
    return np.load(og_path+lang+"/single_test/"+sys.argv[-2]+".npy")

if __name__ == "__main__":
    lang = sys.argv[-1]

    testing_trace = getTestingSuite(lang)
    trace = getTrace(lang)
