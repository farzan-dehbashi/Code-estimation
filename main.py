'''

To run:
python3 main.py [paths to files] # should point to .out files

example: python3 main.py numpy scipy

'''


import numpy as np
import scipy
import tensorflow as tf
import sys
from os import listdir
from os.path import isfile, join
from sklearn.utils import shuffle
from keras.callbacks import EarlyStopping
import time

import parse_output_file
import networks

label_size = 2

train = True

og_path = "power_cunsumption/2k_per_second_"

langs = {"numpy":0,
        "scipy": 1,
        "other": 2}


def balanceDataset(masterDataset, masterLabels):
    min_frames = masterDataset[0].shape[0]

    for i, val in enumerate(masterDataset):
        # print(val.shape)
        if val.shape[0] < min_frames:
            min_frames = val.shape[0]

    X, y = [], []
    for i, val in enumerate(masterDataset):
        frames, labels = shuffle(val, masterLabels[i])
        X.append(frames[:min_frames])
        y.append(labels[:min_frames])

    X = np.concatenate((X))
    y = np.concatenate((y))

    X, y = shuffle(X, y)


    split = int(0.7*X.shape[0])
    val_split = int(0.6*X.shape[0])
    test_X = X[split:]
    test_y = y[split:]
    val_X = X[val_split:split]
    val_y = y[val_split:split]
    X = X[:split]
    y = y[:split]

    # test_y = np.argmax(test_y, axis=1)
    # print(np.count_nonzero(test_y == 0))
    # print(np.count_nonzero(test_y == 1))
    # print(np.count_nonzero(test_y == 2))
    # val_y = np.argmax(val_y, axis=1)
    # print(np.count_nonzero(val_y == 0))
    # print(np.count_nonzero(val_y == 1))
    # print(np.count_nonzero(val_y == 2))
    # y = np.argmax(y, axis=1)
    # print(np.count_nonzero(y == 0))
    # print(np.count_nonzero(y == 1))
    # print(np.count_nonzero(y == 2))
    # exit()

    return X, y, val_X, val_y, test_X, test_y





if __name__ == "__main__":
    masterDataset = []
    masterLabels = []
    if "load" in sys.argv:
        for i, lang in enumerate(sys.argv[1:-1]):
            dataset = np.load(og_path+lang+'/X_'+lang+'_all_200_2classes.npy')
            dataset_y = np.load(og_path+lang+'/y_'+lang+'_all_200_2classes.npy')
            masterDataset.append(dataset)
            masterLabels.append(dataset_y)

        X, y, val_X, val_y, test_X, test_y = balanceDataset(masterDataset, masterLabels)
        dataset = X
    elif 'load_dataset' in sys.argv:
        time = '20200729-233039'
        X = np.load('datasets/'+ time +'_X.npy' )
        y = np.load('datasets/'+ time +'_y.npy' )
        val_X = np.load('datasets/'+ time +'_val_X.npy' )
        val_y = np.load('datasets/'+ time +'_val_y.npy' )
        test_X = np.load('datasets/'+ time +'_test_X.npy' )
        test_y = np.load('datasets/'+ time +'_test_y.npy' )
    else:
        paths = []
        for i, lang in enumerate(sys.argv[1:]):
            print(">>> GETTING ", lang)
            try_paths = [og_path + lang +"/"+f+"/" + lang +"_timestamped_"+f+".out" for f in listdir(og_path + lang +"/") if f[:3] == "try"]

    # print(paths)
            dataset, dataset_tests = parse_output_file.buildDataset(try_paths,lang)
            dataset_y = np.zeros((dataset.shape[0], label_size))
            dataset_y[:,langs[lang]] = dataset_y[:,langs[lang]] + 1

            np.save(og_path+lang+'/X_'+lang+'_all_200_2classes.npy', dataset)
            np.save(og_path+lang+'/y_'+lang+'_all_200_2classes.npy', dataset_y)
            print(lang, " X: ", dataset.shape, " y: ", dataset_y.shape)

            masterDataset.append(dataset)
            masterLabels.append(dataset_y)
        exit()
        X, y, val_X, val_y, test_X, test_y = balanceDataset(masterDataset, masterLabels)

    # masterDataset = np.concatenate((masterDataset))
    # masterLabels = np.concatenate((masterLabels))
    # print(masterDataset.shape)
    # print(masterLabels.shape)
    # exit()
    # dataset = np.expand_dims(dataset, 2)
    X = np.expand_dims(X, 2)
    val_X = np.expand_dims(val_X, 2)
    test_X = np.expand_dims(test_X, 2)

    print(X.shape)
    print(y.shape)
    print(test_X.shape)
    print(test_y.shape)
    print(val_X.shape)
    print(val_y.shape)
    # exit()
    y = np.argmax(y, axis=1)
    test_y = np.argmax(test_y, axis=1)
    val_y = np.argmax(val_y, axis=1)

    if train:
        time_string = time.strftime("%Y%m%d-%H%M%S")
        # mod = networks.bidirectionalLSTM(X)
        mod = networks.CNN_mod(X)

        # es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=20)
        epochs = 10
        # mod.fit(X,y, batch_size=32, epochs=epochs,verbose=1, validation_data=(val_X, val_y)) #, callbacks=[es]
        # mod.save('models/'+str(time_string)+'_epochs'+ str(epochs) +'.h5')

        mod.fit(X,y, batch_size=32, epochs=epochs,verbose=1, validation_data=(val_X, val_y))
        mod.save('models/CNN'+str(time_string)+'.h5')

        np.save('datasets/'+time_string+'_X.npy', X)
        np.save('datasets/'+time_string+'_y.npy', y)
        np.save('datasets/'+time_string+'_val_X.npy', val_X)
        np.save('datasets/'+time_string+'_val_y.npy', val_y)
        np.save('datasets/'+time_string+'_test_X.npy', test_X)
        np.save('datasets/'+time_string+'_test_y.npy', test_y)
    else:
        mod = '20200729-233039.h5'
        mod = networks.bidirectionalLSTM(X, mod)
    preds = mod.predict(test_X)

    # flat_preds = np.argmax(preds, axis=1)
    print(flat_preds)
    flat_test_y = np.argmax(test_y, axis=1)
    print(flat_test_y)
    print("pred = 0: ", np.count_nonzero(flat_preds == 0))
    print("pred = 1: ", np.count_nonzero(flat_preds == 1))
    # print(np.count_nonzero(flat_preds == 2))
    print("ground = 0: ", np.count_nonzero(flat_test_y == 0))
    print("ground = 0: ", np.count_nonzero(flat_test_y == 1))
    # print(np.count_nonzero(flat_test_y == 2))
    diff = flat_test_y - flat_preds
    correct = np.count_nonzero(diff == 0)
    incorrect = np.count_nonzero(diff)
    print("Correct: ", correct, " perc: ", correct / diff.shape[0])
    print("Incorrect: ", incorrect, " perc: ", incorrect / diff.shape[0])
    # print(test_y.shape)
    # print(preds.shape)
    # exit()
