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

# from dtaidistance import dtw
# from dtaidistance import dtw_visualisation as dtwvis



def makePaperFigureFFT(window_fft_diff):
    xtick_vals, xtick_labels = [],[]
    # for i in range(file_data[0].shape[0]):
    #     if i%200 == 0:
    #         xtick_vals.append(i)
    #         xtick_labels.append(round(i/1000,1))

    plt.figure(figsize=(14,5))
    # plt.axvspan(83, 535, color='orange', alpha=0.2)
    # plt.axvspan(671, 1070, color='red', alpha=0.2)
    # plt.axvspan(540, 920, color='blue', alpha=0.2)

    plt.plot(np.arange(window_fft_diff.shape[0]), window_fft_diff, c='b')
    plt.title("FFT Difference - Not Lowest Value of d", fontsize=25)
    plt.ylabel("Value",fontsize=25)
    plt.xlabel("Frequency",fontsize=25)
    # plt.xticks(xtick_vals, xtick_labels,fontsize=22)
    plt.xticks([0],[""])
    plt.yticks(fontsize=22)

    #
    # legend_elements = [ Patch(facecolor='orange', alpha=0.2,label='Flatten'),
    #                     Patch(facecolor='red', alpha=0.2,label='Reshape')]
    # #                     Patch(facecolor='blue',label='Large')]
    # plt.legend(handles=legend_elements, framealpha=1, fontsize=18, loc='upper right')
    plt.show()
    # exit()

def showD(start, end):

    s = file_data[0][start:end]
    t = target

    d, paths = dtw.warping_paths(s, t)
    best_path = dtw.best_path(paths)
    dtwvis.plot_warpingpaths(s, t, paths, best_path)

    plt.title("Query Trace", fontsize=35,y=1.17)
    plt.ylabel("Target Trace", fontsize=35,labelpad=85)
    plt.show()
    exit()


def calculateD(pass_arr):
    start = pass_arr[0]
    end = pass_arr[1]
    # window = power_data[start+100:end+100]
    s = file_data[0][start:end]
    t = target

    # from dtaidistance import dtw
    # from dtaidistance import dtw_visualisation as dtwvis
    #
    # # path = dtw.warping_path(s, t)
    # # dtwvis.plot_warping(s, t, path)
    #
    # d, paths = dtw.warping_paths(s, t)
    # best_path = dtw.best_path(paths)
    # dtwvis.plot_warpingpaths(s, t, paths, best_path)


    # plt.title("Query Trace", fontsize=35,y=1.17)
    # plt.ylabel("Target Trace", fontsize=35,labelpad=85)
    # plt.show()
    # exit()

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
    #         cost = abs(i-j) + abs(s[i-1] - t[j-1])
    #         # cost = abs(s[i-1] - t[j-1])
    #         # take last min from a square box
    #         # last_min = np.min([dtw_matrix[i-1, j], dtw_matrix[i, j-1], dtw_matrix[i-1, j-1]])
    #         dtw_matrix[i, j] = cost #+ last_min
    #
    # cost_mtx = dtw_matrix[1:][:,1:]
    #
    # path = [(0,0)]
    # # for i in range(0,cost_mtx.shape[0]):
    # while path[-1][0] < cost_mtx.shape[0]-1 and path[-1][1] < cost_mtx.shape[0]-1:
    #     cur = path[-1]
    #     dir = np.argmin([cost_mtx[cur[0]+1, cur[1]], cost_mtx[cur[0], cur[1]+1], cost_mtx[cur[0]+1, cur[1]+1]]) #down, right, diag
    #     if dir == 0: path.append((cur[0]+1,cur[1]))
    #     elif dir == 1: path.append((cur[0], cur[1]+1))
    #     else: path.append((cur[0]+1, cur[1]+1))
    #
    # for i in path:
    #     cost_mtx[i[0],i[1]] = 100
    #
    # plt.imshow(cost_mtx, cmap='gray')
    # plt.show()
    # exit()


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


# ******************************************************************************************************************************************************
if mode == "numpy_random_funcs_nosleep_mean_SUBSET_try":
    trace_check_str = "mean"
    func_name = "Mean"
    nums = {0: [4385, 4391],
            1: [5061, 5067],
            2: [4528, 4534],
            3: [4014, 4021],
            4: [4494, 4500],
            5: [4450, 4457],
            6: [4579, 4585],
            7: [4395, 4401],
            8: [4325, 4331],
            9: [4873, 4880]}

if mode == "numpy_random_funcs_nosleep_fft_SUBSET_try":
    trace_check_str = "numpy_fft"
    func_name = "FFT"
    nums = {0: [4799, 6669],
            1: [4395, 6247],
            2: [4549, 6464],
            3: [4741, 6654],
            4: [4684, 6555],
            5: [4540, 6410],
            6: [4114, 5985],
            7: [4519, 6389],
            8: [4634, 6504],
            9: [4281, 6151]}

if mode == "numpy_random_funcs_nosleep_mean_SUBSET_long_try":
    trace_check_str = "mean"
    func_name = "Mean"
    nums = {0: [7890, 8006],
            1: [7531, 7650],
            2: [7177, 7294],
            3: [6582, 6700],
            4: [7190, 7307],
            5: [7116, 7234],
            6: [7152, 7268],
            7: [6965, 7080],
            8: [7278, 7398],
            9: [7030, 7147]}

if mode == "numpy_random_funcs_nosleep_max_SUBSET_long_try":
    trace_check_str = "max"
    func_name = "Amax"
    nums = {0: [6495, 6720],
            1: [6781, 7009],
            2: [7384, 7616],
            3: [7129, 7359],
            4: [7572, 7806],
            5: [6665, 6895],
            6: [7385, 7616],
            7: [7172, 7403],
            8: [6914, 7145],
            9: [6831, 7063]}

if mode == "numpy_random_funcs_nosleep_median_SUBSET_long_try":
    trace_check_str = "median"
    func_name = "Median"
    nums = {0: [6754, 8018],
            1: [7422, 8691],
            2: [7529, 8798],
            3: [7324, 8603],
            4: [7285, 8614],
            5: [7428, 8751],
            6: [7075, 8371],
            7: [7464, 8757],
            8: [7627, 8894],
            9: [7325, 8591]}

# ../../../../../Desktop/Multimeter_SHARE/numpy_mean_grow.csv
# grow_nums = {0: [12687, 12705],
#         1: [17069, 17110],
#         2: [22220, 22293],
#         3: [28370, 28481],
#         4: [35739, 35905],
#         5: [44551, 44780],
#         6: [55035, 55326],
#         7: [67412, 67787]}


# ************************* noseed with sleepers ************************************************************************

if mode == "noseed_numpy_tests_mean_sleep":
    trace_check_str = "mean"
    func_name = "Mean"
    nums = {0: [6612, 6729],
            1: [6668, 6785],
            2: [6653, 6772],
            3: [6572, 6688],
            4: [6868, 6989],
            5: [6931, 7048],
            6: [6934, 7054],
            7: [6887, 7004],
            8: [6987, 7106],
            9: [6758, 6875]}

if mode == "noseed_numpy_tests_median_sleep":
    trace_check_str = "median"
    func_name = "Median"
    nums = {0: [7071, 8991],
            1: [6889, 8300],
            2: [7029, 8915],
            3: [6608, 8417],
            4: [6930, 8313],
            5: [7083, 8912],
            6: [6759, 8235],
            7: [6810, 8818],
            8: [7073, 8760],
            9: [7463, 9015]}

if mode == "noseed_numpy_tests_max_sleep":
    trace_check_str = "max"
    func_name = "Amax"
    nums = {0: [6604, 6836],
            1: [6777, 7009],
            2: [6903, 7142],
            3: [6568, 6798],
            4: [7271, 7511],
            5: [6290, 6522],
            6: [7121, 7360],
            7: [6686, 6926],
            8: [6047, 6276],
            9: [6620, 6849]}


# ==========================================================================================================================
# ========== NO SEED BELOW =================================================================================================
# ==========================================================================================================================
if mode == "noseed_numpy_random_funcs_mean_try":
    trace_check_str = "mean"
    func_name = "Mean"
    nums = {0: [8067, 8183],
            1: [7471, 7587],
            2: [7146, 7262],
            3: [7033, 7149],
            4: [7162, 7278],
            5: [7550, 7666],
            6: [7871, 7989],
            7: [7423, 7538],
            8: [7308, 7424],
            9: [7125, 7240]} #7715, 7831]}

if mode == "noseed_numpy_random_funcs_median_try":
    trace_check_str = "median"
    func_name = "Median"
    nums = {0: [8876, 10809],
            1: [8223, 10100],
            2: [7500, 9288],
            3: [8110, 9507],
            4: [7256, 9021],
            5: [7224, 9054],
            6: [6992, 8864],
            7: [7183, 8871],
            8: [7077, 8475],
            9: [7179, 8872]}

if mode == "noseed_numpy_random_funcs_fft_try":
    trace_check_str = "numpy_fft"
    func_name = "FFT"
    nums = {0: [4692, 6610],
            1: [4245, 6098],
            2: [5062, 6912],
            3: [5383, 7234],
            4: [5146, 7028],
            5: [4580, 6420],
            6: [4387, 6230],
            7: [4489, 6365],
            8: [4977, 6894],
            9: [4600, 6453]}

if mode == "noseed_numpy_random_funcs_max_try":
    trace_check_str = "max"
    func_name = "Amax"
    nums = {0: [7158, 7388],
            1: [7550, 7783],
            2: [6959, 7189],
            3: [6948, 7179],
            4: [7602, 7833],
            5: [6984, 7211],
            6: [7713, 7941],
            7: [6773, 7007],
            8: [6997, 7232],
            9: [7464, 7694]}


# plt.figure(figsize=(14,2))
training_trials = 3
min_d_vals, labels = [], []
all_min_d = []

present_min_d, omitted_min_d = [], []

# Grab the function from a trial
for trial in range(0,10):
    print("Target: ", trial)
    fname = sys.argv[1] + str(trial) + ".csv"
    # fname = '../../../../../Desktop/Multimeter_SHARE/numpy_mean_grow.csv'

    fname_arr = fname.split("/")
    data = parseData(fname)

    f_start = nums[trial][0]
    f_end = nums[trial][1]
    # f_start = grow_nums[trial][0]
    # f_end = grow_nums[trial][1]
    target = data[f_start:f_end]

    # returned_d = calculateD(f_start, (f_start+target.shape[0]), target, printer=True, power_data=data)

    # train logistic regression model
    if trial == training_trials:
        min_d_vals = np.expand_dims(min_d_vals,axis=1)
        labels = np.expand_dims(labels,axis=1)
        # logisticRegr.fit(min_d_vals, labels)
        # kmeans = KMeans(n_clusters=2, random_state=0).fit(min_d_vals)
        # clustering = DBSCAN(eps=0.005, min_samples=2).fit(min_d_vals)
        # print(clustering.labels_)
        # for i, val in enumerate(min_d_vals):
        #     if clustering.labels_[i] == 1: dot_color='r'
        #     else: dot_color='b'
        #     plt.plot(val[0], 0, 'o', color=dot_color)
        # plt.show()
        # exit()
        clf = make_pipeline(StandardScaler(), SGDClassifier(max_iter=1000, tol=1e-3))
        clf.fit(min_d_vals, labels)

    # loop through other traces and compare function to those others
    for try_num in range(0,10):
        # try_num = 9
        fname = sys.argv[1] + str(try_num) + ".csv" #compares to same file

        # ----- These are used with "noseed" to compare to the original experiments
        # fname = '../../../../../Desktop/Multimeter_SHARE/numpy_random_funcs_nosleep_mean_SUBSET_long/numpy_random_funcs_nosleep_mean_SUBSET_long_try'+ str(try_num) + ".csv"
        # fname = '../../../../../Desktop/Multimeter_SHARE/numpy_random_funcs_nosleep_median_SUBSET_long/numpy_random_funcs_nosleep_median_SUBSET_long_try'+ str(try_num) + ".csv"
        # fname = '../../../../../Desktop/Multimeter_SHARE/numpy_random_funcs_nosleep_fft_SUBSET_long/numpy_random_funcs_nosleep_fft_SUBSET_long_try'+ str(try_num) + ".csv"
        # fname = '../../../../../Desktop/Multimeter_SHARE/numpy_random_funcs_nosleep_max_SUBSET_long/numpy_random_funcs_nosleep_max_SUBSET_long_try'+ str(try_num) + ".csv"




        fname_arr = fname.split("/")


        file_data = []
        # for i in sys.argv[1:]:
        file_data.append(parseData(fname))

        if nums is None:
            offset = 6000 #4000
            f_start = 0
            f_end = 0
            for i,val in enumerate(file_data[0][offset:len(file_data[0])//2]):
                if val > 1.5 and f_start == 0:
                    # running.append(3000+i)
                    f_start = offset+i
                if f_start != 0 and val < 1.2 and (offset+i) > (f_start+500):
                    f_end = offset+i
                    break
        else:
            f_start = nums[try_num][0]
            f_end = nums[try_num][1]


        start = f_end+2500
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






        if trial >= training_trials:
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


        # DETECTION_THRESHOLD = 0.02
        # if min_d < DETECTION_THRESHOLD:
        #     alg_detection = True
        # else:
        #     alg_detection = False

        func_df = pd.read_csv(fname_arr[-1][:-9] + "/" + fname_arr[-1][:-4] + '.out', names=['functions'])
        func_arr = [func_str.split(" ")[-1] for func_str in func_df["functions"].values]

        if trace_check_str in func_arr: #func_df["functions"].values.split(" ")[-1]: #numpy_fft, median, mean
            ACTUAL = True
            dot_color_ground = 'b'
            label = 1
            present_min_d.append(min_d)
        else:
            ACTUAL = False
            dot_color_ground = 'r'
            label = 0
            omitted_min_d.append(min_d)

        if trial < training_trials:
            min_d_vals.append(min_d)
            labels.append(label)
        else:
            all_min_d.append(min_d)


        print("ALGORITHM: ", alg_detection, "   ACTUAL: ", ACTUAL, " d: ", min_d)
        if trial >= training_trials:
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



        # if trial >= training_trials:
        #     plt.scatter(min_d, 0, marker='o',s=60, color=dot_color_ground)

        # for i, val in enumerate(s_min_n_idx[:10]):
        #     plt.axvline(x=val, color='g')
        # plt.axvline(x=min_d_idx, color='r')
        # plt.plot(np.arange(file_data[0].shape[0]), file_data[0])
        # plt.show()
        # exit()


np.save("present_min_d-"+mode+".npy", np.array(present_min_d))
np.save("omitted_min_d-"+mode+".npy", np.array(omitted_min_d))

print()
print("********** RESULTS **********")
print("Experiment: ", fname_arr[-1][:-9])
print()
print("TP: ", TP, " TN: ", TN, " FP: ", FP, " FN: ", FN)
print()
precision = TP / (TP+FP)
recall = TP / (TP + FN)
f1 = 2 * ((precision * recall) / (precision + recall))
print("Precision: ", precision)
print("Recall: ", recall)
print("F-Score: ", f1)

result_dict = {"Mode": mode,
                "TP": TP,
                "TN": TN,
                "FP": FP,
                "FN": FN,
                "Pecision": precision,
                "Recall": recall,
                "F1-Score": f1}

with open("results/"+mode+'.txt', 'w') as file:
     file.write(json.dumps(result_dict))

# conf mtx graph
# legend_elements = [Line2D([0], [0], marker='o', color='w', label='TP',markerfacecolor='b', markersize=9),
#                    Line2D([0], [0], marker='o', color='w', label='TN', markerfacecolor='r', markersize=9),
#                    Line2D([0], [0], marker='o', color='w', label='FP', markerfacecolor='g', markersize=9),
#                    Line2D([0], [0], marker='o', color='w', label='FN', markerfacecolor='c', markersize=9)]
#
#
# plt.legend(handles=legend_elements,framealpha=1,ncol=2, fontsize=16,loc='center right')
# plt.title("Numpy Median Detection - Same Array", fontsize=25)
# plt.xlabel("d Value", fontsize=25)
# plt.xticks(fontsize=20)
# plt.yticks([0],[""],fontsize=25)
# plt.ylim(bottom=-0.05, top=0.05)
#
# last = np.sort(np.array(all_min_d))[-1]
# first = np.sort(np.array(all_min_d))[0]
# middle = (last - first) / 2
#
# plt.xlim(left=first*0.9, right= last*1.3)
#
# # plt.axhline(linewidth=2,color='k')
# plt.show()


# ----- graph with "actual" and "classified"
# legend_elements = [Line2D([0], [0], marker='o', color='w', label='Present',markerfacecolor='b', markersize=9),
#                    Line2D([0], [0], marker='o', color='w', label='Omitted', markerfacecolor='r', markersize=9)]
#
# plt.legend(handles=legend_elements,framealpha=1, fontsize=18,loc='center right')
# plt.title("Numpy Function: "+func_name, fontsize=25)
# plt.xlabel("d Value", fontsize=25)
# plt.xticks(fontsize=20)
# # plt.yticks([-0.05,0.05],["Classified", "Actual"],fontsize=25)
# plt.yticks([0],[""])
# plt.ylim(bottom=-0.05, top=0.05)
#
# last = np.sort(np.array(all_min_d))[-1]
# first = np.sort(np.array(all_min_d))[0]
# middle = (last - first) / 2
#
# plt.xlim(left=first*0.9, right= last*1.5)
#
# # plt.axhline(linewidth=2,color='k')
# # plt.show()
# # plt.savefig(fname,bbox_inches='tight', dpi=300)
# exit()
