'''

To run:
python3 main.py [paths to files] # should point to .out files


'''


import numpy as np
import scipy
import tensorflow as tf
import sys
from os import listdir
from os.path import isfile, join

import parse_output_file
import networks

label_size = 3

og_path = "power_cunsumption/time_stamped_results/"

langs = {"numpy":0,
        "scipy": 1,
        "other": 2}


if __name__ == "__main__":
    masterDataset = []
    masterLabels = []

    # maybe hardcode paths in here for numpy and scipy

    paths = []
    for i in sys.argv:
        if i[:4] != 'main':
            try_paths = [og_path+i+"/"+f+"/"+i+"_timestamp_"+f[3]+".out" for f in listdir(og_path+i+"/") if f[:3] == "try"]
            paths = paths + try_paths

    # print(paths)
            dataset, dataset_tests = parse_output_file.buildDataset(paths)
            dataset_y = np.zeros((dataset.shape[0], label_size))
            dataset_y[:,langs[i]] = dataset_y[:,langs[i]] + 1

            masterDataset.append(dataset)
            masterLabels.append(dataset_y)

    masterDataset = np.concatenate((masterDataset))
    masterLabels = np.concatenate((masterLabels))
    print(masterDataset.shape)
    print(masterLabels.shape)
    exit()
    dataset = np.expand_dims(dataset, 2)

    bidir_mod = networks.bidirectionalLSTM(dataset)
    # print(bidir_mod.summary())
    print(dataset.shape)
