import numpy as np
import scipy
import tensorflow as tf
import sys

import parse_output_file
import networks




if __name__ == "__main__":

    # maybe hardcode paths in here for numpy and scipy

    paths = []
    for i in sys.argv:
        if i[:4] != 'main':
            paths.append(i)

    dataset, dataset_tests = parse_output_file.buildDataset(paths)
    dataset = np.expand_dims(dataset, 2)
    
    bidir_mod = networks.bidirectionalLSTM(dataset)
    # print(bidir_mod.summary())
    print(dataset.shape)
