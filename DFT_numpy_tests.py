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



# NEW TODO:
# can different sizes of inputs be used to detect other sizes?


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
    # plt.xticks([0],[""])
    plt.yticks(fontsize=22)

    #
    # legend_elements = [ Patch(facecolor='orange', alpha=0.2,label='Flatten'),
    #                     Patch(facecolor='red', alpha=0.2,label='Reshape')]
    # #                     Patch(facecolor='blue',label='Large')]
    # plt.legend(handles=legend_elements, framealpha=1, fontsize=18, loc='upper right')
    plt.show()
    exit()

# def calculateD(start, end, target, printer=False):
#     # if printer:
#     #     start+=20
#     #     end+=20
#     window = file_data[0][start:end]
#
#     # if printer:
#     #     plt.axvspan(start, end, color='orange', alpha=0.2)
#     #     plt.plot(np.arange(file_data[0].shape[0]), file_data[0])
#     #     plt.show()
#
#
#
#     window_fft = np.fft.fft(window)[1:((end-start)//2)]
#
#     target_fft = np.fft.fft(target)[1:((end-start)//2)]
#
#     window_fft_diff = np.fft.ifft(target_fft/window_fft)[1:(window_fft.shape[0]//2)]
#
#     window_fft_diff = 2.0/window_fft.shape[0] * np.absolute(window_fft_diff[:window_fft.shape[0]//2])
#
#     # difference of medians
#     # diff_med = np.absolute(np.median(window) - np.median(target))
#
#
#     med_abs_val = np.median(np.absolute(window_fft_diff))
#     argmax = np.argmax(window_fft_diff)
#     h_p = np.amax(np.absolute(window_fft_diff))
#     w_p, left, right = getWidth(window_fft_diff, med_abs_val)
#     L_n = np.amax(np.absolute(np.concatenate((window_fft_diff[:left],window_fft_diff[right:]), axis=0))).real
#
#     # amp = np.absolute(np.median(window) - np.median(target))
#     amp = np.mean(np.absolute(window-target))
#     # amp = np.sum(np.absolute(window-target))
#
#     # d = (((w_p * (L_n / h_p)).real) * diff_med) * amp
#     d = ((w_p * (L_n / h_p)).real) * amp
#
#     if printer:
#         # print("W: ", w_p)
#         # print("L: ", L_n)
#         # print("h: ", h_p)
#         # print("amp: ", amp)
#         # print(d)
#         # exit()
#         # a = np.sort(np.absolute(window-target))
#         # print(np.mean(a[-10:]))
#         plt.plot(np.arange(window.shape[0]), window)
#         plt.plot(np.arange(target.shape[0]), target)
#         plt.show()
#         plt.plot(np.arange(window_fft.shape[0]), window_fft)
#         plt.plot(np.arange(target_fft.shape[0]), target_fft)
#         plt.show()
#         # exit()
#         makePaperFigureFFT(window_fft_diff)
#
#     return d


def calculateD(start, end, target, printer=False):
    window = file_data[0][start:end]
    window_fft = np.absolute(np.fft.fft(window)[1:((end-start)//2)])

    target_fft = np.absolute(np.fft.fft(target)[1:((end-start)//2)])

    window_fft_diff = np.absolute(np.fft.ifft(target_fft/window_fft))
    window_fft_diff = 2.0/window_fft.shape[0] * np.absolute(window_fft_diff[:window_fft.shape[0]//2])

    # difference of medians
    # diff_med = np.absolute(np.median(window) - np.median(target))

    med_abs_val = np.median(np.absolute(window_fft_diff))
    argmax = np.argmax(window_fft_diff)
    h_p = np.amax(window_fft_diff)
    w_p, left, right = getWidth(window_fft_diff, med_abs_val)
    L_n = np.amax(np.concatenate((window_fft_diff[:left],window_fft_diff[right:]), axis=0)).real

    # amp = np.absolute(np.median(window) - np.median(target))
    amp = np.mean(np.absolute(window-target))

    # d = (((w_p * (L_n / h_p)).real) * diff_med) * amp
    d = ((w_p * (L_n / h_p)).real) * amp

    # if printer:
        # print("W: ", w_p)
        # print("L: ", L_n)
        # print("h: ", h_p)
        # print("amp: ", amp)
        # print("d: ", d)
        # a = np.sort(np.absolute(window-target))
        # print(np.mean(a[-10:]))
        # plt.plot(np.arange(window.shape[0]), window)
        # plt.plot(np.arange(target.shape[0]), target)
        # plt.show()
        # makePaperFigureFFT(window_fft_diff)
        # exit()

    return d


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


# ************************************************** Function start in numpy_tests ************************************************************
# # seed_numpy_tests_mean_sleep/seed_numpy_tests_mean_sleep_try
# nums = {0: [6610, 6740],
#         1: [6427, 6557],
#         2: [7090, 7224],
#         3: [6648, 6743],
#         4: [6458, 6588],
#         5: [5948, 6078],
#         6: [7190, 7320],
#         7: [7420, 7551],
#         8: [6627, 6758],
#         9: [6414, 6547]}
#
# func_start = {0: 8441,
#             1: 8256,
#             2: 8923,
#             3: 8478,
#             4: 8286,
#             5: 7777,
#             6: 9019,
#             7: 9250,
#             8: 8456,
#             9: 8245}

# # seed_numpy_tests_median_sleep/seed_numpy_tests_median_sleep_try
# nums = {0: [6648, 7954],
#         1: [6860, 8169],
#         2: [6473, 7779],
#         3: [7037, 8343],
#         4: [6810, 8116],
#         5: [6442, 7745],
#         6: [6525, 7833],
#         7: [6574, 7934],
#         8: [7753, 9064],
#         9: [6489, 7799]}
#
# func_start = {0: 11476, #THIS IS THE BEGINNING OF THE FUNCTION
#             1: 11689,
#             2: 11298,
#             3: 11863,
#             4: 11634,
#             5: 11262,
#             6: 11353,
#             7: 11456,
#             8: 12580,
#             9: 11316}

# seed_numpy_tests_fft_sleep/seed_numpy_tests_fft_sleep_try
# nums = {0: [4252, 6106],
#         1: [4503, 6350],
#         2: [4210, 6064],
#         3: [4005, 5862],
#         4: [4155, 6012],
#         5: [4549, 6404],
#         6: [4011, 5867],
#         7: [5047, 6901],
#         8: [4047, 5902],
#         9: [4495, 6345]}
# #
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

# func_start = {0: 7790, #THIS IS THE BEGINNING OF THE FFT FUNCTION
#             1: 8038,
#             2: 7745,
#             3: 7546,
#             4: 7696,
#             5: 8089,
#             6: 7555,
#             7: 8593,
#             8: 7589,
#             9: 8030}

# # seed_numpy_tests_max_sleep/seed_numpy_tests_max_sleep_try
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
func_start = {0: 13642, # ENDING
            1: 14196,
            2: 13404,
            3: 13474,
            4: 14088,
            5: 13716,
            6: 14086,
            7: 13878,
            8: 13946,
            9: 13946}



# ******************************************************************************************************************************************************
# numpy_random_funcs_nosleep_mean_SUBSET_try
# nums = {0: [4385, 4391],
#         1: [5061, 5067],
#         2: [4528, 4534],
#         3: [4014, 4021],
#         4: [4494, 4500],
#         5: [4450, 4457],
#         6: [4579, 4585],
#         7: [4395, 4401],
#         8: [4325, 4331],
#         9: [4873, 4880]}

# numpy_random_funcs_nosleep_fft_SUBSET_try
# nums = {0: [4489, 6351],
#         1: [4535, 6397],
#         2: [4549, 6476],
#         3: [4740, 6662],
#         4: [4492, 6507],
#         5: [4540, 6419],
#         6: [4114, 5992],
#         7: [4256, 6397],
#         8: [4633, 6514],
#         9: [4185, 6160]}

# numpy_random_funcs_nosleep_mean_SUBSET_long_try
# nums = {0: [7890, 8006],
#         1: [7531, 7650],
#         2: [7177, 7294],
#         3: [6582, 6700],
#         4: [7190, 7307],
#         5: [7116, 7234],
#         6: [7152, 7268],
#         7: [6965, 7080],
#         8: [7278, 7398],
#         9: [7030, 7147]}

# numpy_random_funcs_nosleep_max_SUBSET_long_try
# nums = {0: [6489, 6725],
#         1: [6776, 7014],
#         2: [7384, 7616],
#         3: [7129, 7359],
#         4: [7572, 7806],
#         5: [6665, 6895],
#         6: [7385, 7616],
#         7: [7172, 7403],
#         8: [6914, 7145],
#         9: [6831, 7063]}

# numpy_random_funcs_nosleep_median_SUBSET_long_try
# nums = {0: [6754, 8018],
#         1: [7422, 8691],
#         2: [7529, 8798],
#         3: [7324, 8603],
#         4: [7285, 8614],
#         5: [7428, 8751],
#         6: [7075, 8371],
#         7: [7464, 8757],
#         8: [7627, 8894],
#         9: [7325, 8591]}

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

# noseed_numpy_tests_mean_sleep
# nums = {0: [6612, 6729],
#         1: [6668, 6785],
#         2: [6653, 6772],
#         3: [6572, 6688],
#         4: [6868, 6989],
#         5: [6931, 7048],
#         6: [6934, 7054],
#         7: [6887, 7004],
#         8: [6987, 7106],
#         9: [6758, 6875]}

# noseed_numpy_tests_median_sleep
# nums = {0: [7071, 8991],
#         1: [6889, 8300],
#         2: [7029, 8915],
#         3: [6608, 8417],
#         4: [6930, 8313],
#         5: [7083, 8912],
#         6: [6759, 8235],
#         7: [6810, 8818],
#         8: [7073, 8760],
#         9: [7463, 9015]}

# noseed_numpy_tests_max_sleep
# nums = {0: [6604, 6836],
#         1: [6777, 7009],
#         2: [6903, 7142],
#         3: [6568, 6798],
#         4: [7271, 7511],
#         5: [6290, 6522],
#         6: [7121, 7360],
#         7: [6686, 6926],
#         8: [6047, 6276],
#         9: [6620, 6849]}


# ==========================================================================================================================
# ========== NO SEED BELOW =================================================================================================
# ==========================================================================================================================
# noseed_numpy_random_funcs_mean_try
# nums = {0: [8067, 8183],
#         1: [7471, 7587],
#         2: [7146, 7262],
#         3: [7033, 7149],
#         4: [7162, 7278],
#         5: [7550, 7666],
#         6: [7871, 7989],
#         7: [7423, 7538],
#         8: [7308, 7424],
#         9: [7715, 7831]}

# noseed_numpy_random_funcs_median_try
# nums = {0: [7613, 9355],
#         1: [7460, 9182],
#         2: [7500, 9288],
#         3: [8110, 9507],
#         4: [7256, 9021],
#         5: [7224, 9054],
#         6: [6992, 8864],
#         7: [7183, 8871],
#         8: [7077, 8475],
#         9: [7179, 8872]}

# noseed_numpy_random_funcs_max_try
# nums = {0: [7158, 7388],
#         1: [7550, 7783],
#         2: [6959, 7189],
#         3: [6948, 7179],
#         4: [7602, 7833],
#         5: [6984, 7211],
#         6: [7713, 7941],
#         7: [6773, 7007],
#         8: [6997, 7232],
#         9: [7464, 7694]}

# noseed_numpy_random_funcs_fft_try
# nums = {0: [4692, 6610],
#         1: [4245, 6098],
#         2: [5062, 6912],
#         3: [5383, 7234],
#         4: [5146, 7028],
#         5: [4580, 6420],
#         6: [4387, 6230],
#         7: [4489, 6365],
#         8: [4977, 6894],
#         9: [4600, 6453]}


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
    # fname = '../../../../../Desktop/Multimeter_SHARE/numpy_grow_max.csv'
    # fname = '../../../../../Desktop/Multimeter_SHARE/numpy_random_funcs_nosleep_max_SUBSET_long/numpy_random_funcs_nosleep_max_SUBSET_long_try'+ str(trial) + ".csv"
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


        # for i, val in enumerate(running[:10]):
        #     plt.axvline(x=val, color='g')
        # print(f_start, " - ", f_end)
        # plt.axvline(x=f_start, color='g')
        # plt.axvline(x=f_end, color='r')
        # plt.plot(np.arange(file_data[0].shape[0]), file_data[0])
        # plt.show()
        # exit()

        # numpy_tests:
        # try1: 9212 - 9362
        # try2: 8868 - 9000
        # try3: 9472 - 9604

        # numpy_random_funcs - np.mean
        # try1: 23674 - 23794 (works)
        # try2: 9360 - 9482 (works)
        # try3: 7724 - 7849 (works)

        # f_start = 21920
        # f_end = 21927

        # target = file_data[0][f_start:f_end] #uncomment for each individual trial matching to itself

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
        while end < file_data[0].shape[0]:
        # while end < 12500:
            # if np.amax(file_data[0][start:end]) > 1.5:
            count += 1
            d = calculateD(start, end, target)

            min_n.append(d)
            min_n_idx.append(start)
            # min_n_amp.append(amp)

            if d < min_d:
                min_d_idx = end #start
                min_d = d

            start+=step
            end+=step

        returned_d = calculateD(min_d_idx, (min_d_idx+target.shape[0]), target, printer=True)


        min_n = np.array(min_n)
        min_n_idx = np.array(min_n_idx)
        # min_n_amp = np.array(min_n_amp)

        a = min_n.argsort()
        s_min_n = min_n[a]
        s_min_n_idx = min_n_idx[a]
        # s_min_n_amp = min_n_amp[a]

        # makePaperFigure(file_data, min_d_idx, (end-start), f_start, f_end, target_extended)

        # print("Min d: ", min_d, " IDX: ", min_d_idx)

        # print(s_min_n[:10])
        # print(s_min_n_idx[:10])
        # print(s_min_n_amp[:10])

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


        #
        # # DETECTION_THRESHOLD = 0.02
        # # if min_d < DETECTION_THRESHOLD:
        # #     alg_detection = True
        # # else:
        # #     alg_detection = False
        #
        # func_df = pd.read_csv(fname_arr[-1][:-9] + "/" + fname_arr[-1][:-4] + '.out', names=['functions'])
        #
        # func_arr = [func_str.split(" ")[-1] for func_str in func_df["functions"].values]
        #
        # if "numpy_fft" in func_arr: #func_df["functions"].values.split(" ")[-1]: #numpy_fft, median, mean
        #     ACTUAL = True
        #     dot_color_ground = 'b'
        #     label = 1
        # else:
        #     ACTUAL = False
        #     dot_color_ground = 'r'
        #     label = 0

        # if trial < training_trials:
        #     min_d_vals.append(min_d)
        #     labels.append(label)
        # else:
        #     all_min_d.append(min_d)


        # if not ACTUAL:
        #     # (((w_p * (L_n / h_p)).real) * diff_med) * amp
        #     returned_d = calculateD(min_d_idx, (min_d_idx+target.shape[0]), target)
            # plt.plot(np.arange(target.shape[0]), file_data[0][min_d_idx:(min_d_idx+target.shape[0])])
            # plt.plot(np.arange(target.shape[0]), target)
            # plt.show()
        # exit()

        # print("ALGORITHM: ", alg_detection, "   ACTUAL: ", ACTUAL, " d: ", min_d)
        print("Accuracy: ", alg_detection, " - difference: ", time_diff)




        # if trial >= training_trials:
            # plt.scatter(min_d, 0.05, marker='o',s=90, color=dot_color_ground)
            # plt.scatter(min_d, -0.05, marker='o',s=90, color=dot_color_classify)
            # plt.scatter(min_d, 0, marker='o',s=60, color=dot_color_ground)
            # plt.scatter(min_d, 0, marker='o',s=60, color=dot_color_result)

        # for i, val in enumerate(s_min_n_idx[:10]):
        #     plt.axvline(x=val, color='g')
        # plt.axvline(x=min_d_idx, color='r')
        # plt.plot(np.arange(file_data[0].shape[0]), file_data[0])
        # plt.show()
        # exit()



print()
print("********** RESULTS **********")
print("Experiment: ", fname_arr[-1][:-9])
print()
print("Accurate: ", accurate)
print("Inaccurate: ", inaccurate)
try:
    print("AVG time diff: ", total_time_diff / accurate)
except:
    print("Accurate is zero")

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
# plt.title("Numpy Function: FFT", fontsize=25)
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
# plt.xlim(left=first*0.9, right= last*1.3)
#
# # plt.axhline(linewidth=2,color='k')
# plt.show()
# exit()
