import numpy as np
import sys
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from sklearn.linear_model import LogisticRegression
logisticRegr = LogisticRegression()

from sklearn.cluster import KMeans

from sklearn.cluster import DBSCAN

from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

from fastdtw import fastdtw
from scipy.spatial.distance import euclidean

import multiprocessing as mp
from multiprocessing import Pool


def calculateD(pass_arr):
    start = pass_arr[0]
    end = pass_arr[1]
    # window = power_data[start+100:end+100]
    s = file_data[0][start:end]
    t = target

    d, path = fastdtw(s, t, dist=euclidean)

    # n, m = len(s), len(t)
    # dtw_matrix = np.zeros((n+1, m+1))
    # for i in range(n+1):
    #     for j in range(m+1):
    #         dtw_matrix[i, j] = np.inf
    # dtw_matrix[0, 0] = 0
    #
    # for i in range(1, n+1):
    #     for j in range(1, m+1):
    #         cost = abs(s[i-1] - t[j-1])
    #         # take last min from a square box
    #         last_min = np.min([dtw_matrix[i-1, j], dtw_matrix[i, j-1], dtw_matrix[i-1, j-1]])
    #         dtw_matrix[i, j] = cost + last_min
    #
    # cost = dtw_matrix[-1][-1]
    return {"d": d, "start": start, "end": end}


def parseData(path):
    # power = np.load(path)

    #
    path = path[:-4]
    data = pd.read_csv(path + '.csv')
    data = data[16:-10]
    data = data.rename(columns={'Address': "time", 'USB0::0x2A8D::0x0101::MY59001058::0::INSTR': "power"})
    data["power"] = pd.to_numeric(data["power"], downcast="float")
    power = np.array(data['power'], dtype=np.float32)*5
    power_time = np.array(data['time'], dtype=np.float32)

    # power = np.concatenate((power[:7500], power[13500:15000]),axis=0)
    return power

def makePaperFigure(file_data, min_d_idx, target_len, f_start, f_end):
    xtick_vals, xtick_labels = [],[]

    # power = np.concatenate((file_data[0][f_start-300:f_end+300], file_data[0][min_d_idx-300:min_d_idx+(target_len+300)]), axis=0)
    # color_start = 900+target_len

    power = file_data[0][f_start-300:-6000]#np.concatenate((file_data[0][f_start-300:f_end+300], file_data[0][min_d_idx-300:min_d_idx+(target_len+300)]), axis=0)
    color_start = min_d_idx - (f_start-300)


    for i in range(power.shape[0]):
        if i%2000 == 0:
            xtick_vals.append(i)
            xtick_labels.append(round(i/1000,1))

    plt.figure(figsize=(14,5))
    plt.axvspan(300, target_len+300, color='blue', alpha=0.2)
    plt.axvspan(color_start, color_start+target_len, color='red', alpha=0.2)
    # plt.axvspan(540, 920, color='blue', alpha=0.2)

    plt.plot(np.arange(power.shape[0]), power, c='b')
    # plt.axvline(x=min_d_idx, color='r')
    plt.title("Minimum d Value: Numpy Amax", fontsize=25)
    plt.ylabel("Watts",fontsize=25)
    plt.xlabel("Time (s)",fontsize=25)
    plt.xticks(xtick_vals, xtick_labels,fontsize=22)
    plt.yticks(fontsize=22)
    plt.ylim(top=np.amax(power)+0.4)

    #
    legend_elements = [ Patch(facecolor='blue', alpha=0.2,label='Target'),
                        Patch(facecolor='red', alpha=0.2,label='Minimum d Value')]
    # #                     Patch(facecolor='blue',label='Large')]
    plt.legend(handles=legend_elements, framealpha=1, fontsize=18, loc='upper right')
    plt.show()
    exit()

def getWidth(window_fft_diff, med_abs_val):
    argmax = np.argmax(window_fft_diff)
    left, right = argmax, argmax
    while left > 0 and window_fft_diff[left] > med_abs_val:
        left-=1
    while right < window_fft_diff.shape[0] and window_fft_diff[right] > med_abs_val:
        right+=1
    return right - left, left, right


TP, TN, FP, FN = 0,0,0,0
nums = None

mode = sys.argv[-1].split("/")[-1]

# ************************************************* GROW BELOW *************************************************

# ../../../../../Desktop/Multimeter_SHARE/numpy_grow_mean.csv
if mode == "numpy_random_funcs_nosleep_mean_SUBSET_long_try":
    grow_file = "traces/numpy_grow_mean.csv"
    trace_check_str = "mean"
    grow_nums = {0: [12687, 12705],
            1: [17069, 17110],
            2: [22220, 22293],
            3: [28370, 28481],
            4: [35739, 35905],
            5: [44551, 44780],
            6: [55035, 55326],
            7: [67412, 67787]}

# ../../../../../Desktop/Multimeter_SHARE/numpy_grow_median.csv
if mode == "numpy_random_funcs_nosleep_median_SUBSET_long_try":
    grow_file = "traces/numpy_grow_median.csv"
    trace_check_str = "median"
    grow_nums = {0: [6389, 6452],
                1: [10279, 10531],
                2: [14894, 15536],
                3: [20647, 21504],
                4: [27582, 29129],
                5: [36384, 38775],
                6: [47421, 50738],
                7: [60989, 64901]}

# ../../../../../Desktop/Multimeter_SHARE/numpy_grow_fft.csv
if mode == "numpy_random_funcs_nosleep_fft_SUBSET_try":
    grow_file = "traces/numpy_grow_fft.csv"
    trace_check_str = "numpy_fft"
    grow_nums = {0: [5663, 5854],
                1: [9296, 9713],
                2: [13195, 14100],
                3: [17656, 19541],
                4: [23254, 27453],
                5: [31455, 40120],
                6: [44756, 63714],
                7: [69583, 113425]}



# ../../../../../Desktop/Multimeter_SHARE/numpy_grow_max.csv
if mode == "numpy_random_funcs_nosleep_max_SUBSET_long_try":
    grow_file = "traces/numpy_grow_max.csv"
    trace_check_str = "max"
    grow_nums = {0: [6561, 6571],
                 1: [10402, 10440],
                 2: [14804, 14889],
                 3: [20002, 20150],
                 4: [26227, 26458],
                 5: [33712, 34043],
                 6: [42689, 43143],
                 7: [53398, 53986]}


# plt.figure(figsize=(14,2))
training_nums = 4



all_min_d = []
# Grab the function from a trial
for trial in range(0,len(grow_nums.keys())):
    print("Target: ", trial)
    # fname = sys.argv[1] + str(trial) + ".csv"
    fname = grow_file
    fname_arr = fname.split("/")
    data = parseData(fname)

    # f_start = nums[trial][0]
    # f_end = nums[trial][1]
    f_start = grow_nums[trial][0]
    f_end = grow_nums[trial][1]
    target = data[f_start:f_end]

    min_d_vals, labels = [], []
    # loop through other traces and compare function to those others
    for try_num in range(0,10):

        # train logistic regression model
        if try_num == training_nums:
            min_d_vals = np.expand_dims(min_d_vals,axis=1)
            labels = np.expand_dims(labels,axis=1)
            clf = make_pipeline(StandardScaler(), SGDClassifier(max_iter=1000, tol=1e-3))
            clf.fit(min_d_vals, labels)


        # try_num = 9
        fname = sys.argv[1] + str(try_num) + ".csv"
        fname_arr = fname.split("/")


        file_data = []
        # for i in sys.argv[1:]:
        file_data.append(parseData(fname))

        # if nums is None:
        #     offset = 6000 #4000
        #     f_start = 0
        #     f_end = 0
        #     for i,val in enumerate(file_data[0][offset:len(file_data[0])//2]):
        #         if val > 1.5 and f_start == 0:
        #             # running.append(3000+i)
        #             f_start = offset+i
        #         if f_start != 0 and val < 1.2 and (offset+i) > (f_start+500):
        #             f_end = offset+i
        #             break
        # else:
        #     f_start = nums[try_num][0]
        #     f_end = nums[try_num][1]



        start = file_data[0].shape[0]//2 #f_end+1000
        end = start + target.shape[0]
        step = 1
        min_d = 10000000000
        min_d_idx = 0

        min_med = 1000
        min_med_idx = 0




        min_n = []
        min_n_idx = []
        min_n_amp = []
        count = 0


        pass_arr = []
        while end < file_data[0].shape[0]:
            pass_arr.append([start, end])
            start+=step
            end+=step

        cores = 40
        chunksize = 1
        with Pool(processes=cores) as pool:
            results_d = pool.map(calculateD, pass_arr, chunksize)


        for i, result in enumerate(results_d):
            if result["d"] < min_d:
                min_d_idx = result["start"]
                min_d = result["d"]


        if try_num >= training_nums:
            # ans = logisticRegr.predict(np.array([[min_d]]))[0]
            ans = clf.predict(np.array([[min_d]]))[0]
            if ans == 1:
                alg_detection = True
                dot_color_classify = 'b'
            else:
                alg_detection = False
                dot_color_classify = 'r'
        else:
            alg_detection = "training"



        func_df = pd.read_csv(fname_arr[-1][:-9] + "/" + fname_arr[-1][:-4] + '.out', names=['functions'])

        func_arr = [func_str.split(" ")[-1] for func_str in func_df["functions"].values]

        if trace_check_str in func_arr: #func_df["functions"].values.split(" ")[-1]: #numpy_fft, median, mean
            ACTUAL = True
            dot_color_ground = 'b'
            label = 1
        else:
            ACTUAL = False
            dot_color_ground = 'r'
            label = 0

        if try_num < training_nums:
            min_d_vals.append(min_d)
            labels.append(label)
        else:
            all_min_d.append(min_d)

        print("ALGORITHM: ", alg_detection, "   ACTUAL: ", ACTUAL, " d: ", min_d)
        # if try_num >= training_nums:
        if alg_detection != 'training':
            if alg_detection and ACTUAL:
                TP+=1
                dot_color_result = 'b'
            elif alg_detection and not ACTUAL:
                FP+=1
                dot_color_result = 'g'
            elif ACTUAL and not alg_detection:
                FN+=1
                dot_color_result = 'c'
            else:
                TN+=1
                dot_color_result = 'r'


precision = TP / (TP+FP)
recall = TP / (TP + FN)
f1 = 2 * ((precision * recall) / (precision + recall))

print()
print("********** RESULTS **********")
print("Experiment: ", fname_arr[-1][:-9])
print()
print("TP: ", TP, " TN: ", TN, " FP: ", FP, " FN: ", FN)
print()
print("Precision: ", precision)
print("Recall: ", recall)
print("F-Score: ", f1)
