import numpy as np
import sys
import json
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

def makePaperFigure(file_data, min_d_idx, target_len, f_start, f_end, target_extended):
    xtick_vals, xtick_labels = [],[]

    # power = np.concatenate((target_extended, file_data[0][min_d_idx-300:min_d_idx+(target_len+600)]), axis=0)     # FOR NP.AMAX
    # power = np.concatenate((target_extended, file_data[0][7600:-1000]), axis=0)
    # min_d_idx = min_d_idx-7600
    # color_start = min_d_idx+target_extended.shape[0]

    power = np.concatenate((file_data[0][f_start-300:f_end+300], file_data[0][min_d_idx-300:min_d_idx+(target_len+300)]), axis=0)
    color_start = 900+target_len

    # power = np.concatenate((file_data[0][f_start-300:f_end+300], file_data[0][min_d_idx-300:min_d_idx+(target_len+300)]), axis=0)
    # power = file_data[0][f_start-300:-4000]#np.concatenate((file_data[0][f_start-300:f_end+300], file_data[0][min_d_idx-300:min_d_idx+(target_len+300)]), axis=0)
    # color_start = min_d_idx - (f_start-300)


    for i in range(power.shape[0]):
        if i%2000 == 0:
            xtick_vals.append(i)
            xtick_labels.append(round(i/1000,1))

    plt.figure(figsize=(14,5))
    plt.axvspan(300, target_len+300, color='blue', alpha=0.2)
    plt.axvspan(color_start, color_start+target_len, color='red', alpha=0.2)
    # plt.axvspan(color_start-target_len, color_start, color='red', alpha=0.2)
    # plt.axvspan(540, 920, color='blue', alpha=0.2)

    plt.plot(np.arange(power.shape[0]), power, c='b')
    # plt.axvline(x=min_d_idx, color='r')
    plt.title("Minimum d Value: Numpy Amax", fontsize=25)
    plt.ylabel("Watts",fontsize=25)
    plt.xlabel("Time (s)",fontsize=25)
    plt.xticks(xtick_vals, xtick_labels,fontsize=22)
    plt.yticks(fontsize=22)
    plt.ylim(top=np.amax(power)+0.4)
    plt.xlim(left=0.0)

    #
    legend_elements = [ Patch(facecolor='blue', alpha=0.2,label='Target'),
                        Patch(facecolor='red', alpha=0.2,label='Minimum d Value')]
    # #                     Patch(facecolor='blue',label='Large')]
    plt.legend(handles=legend_elements, framealpha=1, fontsize=18, loc='upper right')
    plt.show()
    # exit()

def getWidth(window_fft_diff, med_abs_val):
    argmax = np.argmax(window_fft_diff)
    window_fft_diff = np.absolute(window_fft_diff)
    left, right = argmax, argmax
    while left > 0 and window_fft_diff[left] > med_abs_val:
        left-=1
    while right < window_fft_diff.shape[0] and window_fft_diff[right] > med_abs_val:
        right+=1
    return right - left, left, right


TP, TN, FP, FN = 0,0,0,0
nums = None


mode = sys.argv[-1].split("/")[-1]

# ************************************************** Function start in numpy_tests ************************************************************
# # seed_numpy_tests_mean_sleep/seed_numpy_tests_mean_sleep_try
if mode == "seed_numpy_tests_mean_sleep_try":
    nums = {0: [6610, 6740],
            1: [6427, 6557],
            2: [7090, 7224],
            3: [6648, 6743],
            4: [6458, 6588],
            5: [5948, 6078],
            6: [7190, 7320],
            7: [7420, 7551],
            8: [6627, 6758],
            9: [6414, 6547]}

    func_start = {0: 8441,
                1: 8256,
                2: 8923,
                3: 8478,
                4: 8286,
                5: 7777,
                6: 9019,
                7: 9250,
                8: 8456,
                9: 8245}

# # seed_numpy_tests_median_sleep/seed_numpy_tests_median_sleep_try
if mode == "seed_numpy_tests_median_sleep_try":
    nums = {0: [6648, 7954],
            1: [6860, 8169],
            2: [6473, 7779],
            3: [7037, 8343],
            4: [6810, 8116],
            5: [6442, 7745],
            6: [6525, 7833],
            7: [6574, 7934],
            8: [7753, 9064],
            9: [6489, 7799]}

    func_start = {0: 11476, #THIS IS THE BEGINNING OF THE FUNCTION
                1: 11689,
                2: 11298,
                3: 11863,
                4: 11634,
                5: 11262,
                6: 11353,
                7: 11456,
                8: 12580,
                9: 11316}

# seed_numpy_tests_fft_sleep/seed_numpy_tests_fft_sleep_try
if mode == "seed_numpy_tests_fft_sleep_try":
    nums = {0: [4252, 6106],
            1: [4503, 6350],
            2: [4210, 6064],
            3: [4005, 5862],
            4: [4155, 6012],
            5: [4549, 6404],
            6: [4011, 5867],
            7: [5047, 6901],
            8: [4047, 5902],
            9: [4495, 6345]}
    #
    # func_start = {0: 9165, #THIS IS THE END OF THE FFT FUNCTION
    #             1: 9408,
    #             2: 9123,
    #             3: 8920,
    #             4: 9071,
    #             5: 9465,
    #             6: 8930,
    #             7: 9961,
    #             8: 8964,
    #             9: 9404}

    func_start = {0: 7790, #THIS IS THE BEGINNING OF THE FFT FUNCTION
                1: 8038,
                2: 7745,
                3: 7546,
                4: 7696,
                5: 8089,
                6: 7555,
                7: 8593,
                8: 7589,
                9: 8030}

# # seed_numpy_tests_max_sleep/seed_numpy_tests_max_sleep_try
if mode == "seed_numpy_tests_max_sleep_try":
    nums = {0: [6633, 6880],
            1: [7222, 7465], #THESE TRACES FOR MAX ARE WEIRD... IDK WHY
            2: [6428, 6670],
            3: [6500, 6743],
            4: [7115, 7355],
            5: [6742, 6982],
            6: [7114, 7353],
            7: [6907, 7147],
            8: [6977, 7217],
            9: [6978, 7219]}
    # #
    # func_start = {0: 13642, # ENDING
    #             1: 14196,
    #             2: 13404,
    #             3: 13474,
    #             4: 14088,
    #             5: 13716,
    #             6: 14086,
    #             7: 13878,
    #             8: 13946,
    #             9: 13946}
    func_start = {0: 6633, # BEGINNING
                1: 7224,
                2: 6428,
                3: 6503,
                4: 7117,
                5: 6742,
                6: 7114,
                7: 6908,
                8: 6977,
                9: 6977}



# plt.figure(figsize=(14,2))
training_trials = 3
min_d_vals, labels = [], []
all_min_d = []
total_time_diff = 0
accurate, inaccurate = 0, 0
# Grab the function from a trial
for trial in range(0,10):
    print("Target: ", trial)
    fname = sys.argv[1] + str(trial) + ".csv"
    fname_arr = fname.split("/")
    data = parseData(fname)

    f_start = nums[trial][0]
    f_end = nums[trial][1]
    # f_start = grow_nums[trial][0]
    # f_end = grow_nums[trial][1]
    target = data[f_start:f_end]
    target_extended = data[(f_start-300):(f_end+300)]



    # loop through other traces and compare function to those others
    for try_num in range(0,10):
        # try_num = 9
        fname = sys.argv[1] + str(try_num) + ".csv"
        fname_arr = fname.split("/")


        file_data = []
        # for i in sys.argv[1:]:
        file_data.append(parseData(fname))


        start = f_end+300
        end = start + target.shape[0]
        step = 1
        min_d = 10
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

        time_diff = np.absolute(min_d_idx - func_start[try_num])


        if time_diff < 150:
            alg_detection = True
            dot_color_classify = 'b'
            accurate += 1
            total_time_diff += time_diff
        else:
            alg_detection = False
            dot_color_classify = 'r'
            inaccurate += 1


        print("Accuracy: ", alg_detection, " - difference: ", time_diff)



print()
print("********** RESULTS **********")
print("Experiment: ", fname_arr[-1][:-9])
print()
print("Accurate: ", accurate)
print("Inaccurate: ", inaccurate)
try:
    avg_time_diff = total_time_diff / accurate
    print("AVG time diff: ", avg_time_diff)
except:
    avg_time_diff = 100000
    print("Accurate is zero")


result_dict = {"Mode": mode,
                "Accurate": accurate,
                "Inaccurate": inaccurate,
                "AVG_Time_Diff": avg_time_diff}

with open("results/"+mode+'.txt', 'w') as file:
     file.write(json.dumps(exDict))
