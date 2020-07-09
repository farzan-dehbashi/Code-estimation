import pandas as pd
import numpy as np
import sys
import random
from matplotlib import pyplot as plt
from matplotlib.font_manager import FontProperties


def subplots(data_arrs):

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1)
    fig.subplots_adjust(hspace=0.5)

    dt = 0.01
    t = np.arange(0, 30, dt)

    min_y, max_y = [], []
    for i, val in enumerate(data_arrs):
        min_y.append(np.amin(val[0]))
        max_y.append(np.amax(val[0]))

    min_y = np.amin(np.array(min_y))
    max_y = np.amax(np.array(max_y))

    colors = []
    for _ in data_arrs[0][1]:
        colors.append((random.uniform(0, 1), random.uniform(0, 1), random.uniform(0, 1)))


    ax1.set_title('Numpy')
    ax1.plot(np.arange(data_arrs[0][0].shape[0]), data_arrs[0][0])
    ax1.set_ylim(min_y, max_y)
    for i, xc in enumerate(data_arrs[0][1]):
        ax1.axvline(x=xc, color=colors[i], label=data_arrs[0][2][i])

    # plt.xticks(tick, tick_labels)
    # plt.xlabel("Minutes")
    # plt.ylabel("Watts")
    ax1.legend(bbox_to_anchor=(1, 1), loc='upper left', ncol=1)


    # ax1.set_ylim(min_y, max_y)
    ax1.set_xticks(data_arrs[0][3])
    ax1.set_xticklabels(data_arrs[0][4])
    ax1.set_xlabel('time (minutes)')
    ax1.set_ylabel('Watts')
    # ax1.grid(True)

    # cxy, f = ax2.csd(s1, s2, 256, 1. / dt)
    ax2.plot(np.arange(data_arrs[1][0].shape[0]), data_arrs[1][0])
    ax2.set_ylim(min_y, max_y)
    for i, xc in enumerate(data_arrs[1][1]):
        ax2.axvline(x=xc, color=colors[i], label=data_arrs[1][2][i])

    ax2.set_xticks(data_arrs[1][3])
    ax2.set_xticklabels(data_arrs[1][4])
    ax2.set_xlabel('time (minutes)')
    ax2.set_ylabel('Watts')

    ax3.plot(np.arange(data_arrs[1][0].shape[0]), data_arrs[1][0])
    ax3.set_ylim(min_y, max_y)
    for i, xc in enumerate(data_arrs[1][1]):
        ax3.axvline(x=xc, color=colors[i], label=data_arrs[1][2][i])

    ax3.set_xticks(data_arrs[1][3])
    ax3.set_xticklabels(data_arrs[1][4])
    ax3.set_xlabel('time (minutes)')
    ax3.set_ylabel('Watts')

    # ax3.plot(x, np3)
    # ax3.set_ylim(min_y, max_y)
    # ax3.set_xticks(tick)
    # ax3.set_xticklabels(tick_labels)
    # ax3.set_xlabel('time (minutes)')
    # ax3.set_ylabel('Watts')
    plt.show()


def plotExperiment(test_start, line_labels,path):
    path = path[:-4]
    data = pd.read_csv(path + '.csv')
    data = data[16:-10]
    data = data.rename(columns={'Address': "time", 'USB0::0x2A8D::0x0101::MY59001058::0::INSTR': "power"})
    data["power"] = pd.to_numeric(data["power"], downcast="float")
    np1 = np.array(data['power'])*5
    np1_time = np.array(data['time'])


    cut = 0
    start_time = 0
    for i, val in enumerate(np1):
        if i == 0:
            time = np1_time[i].split(" ")[1]
            first_time_powermeter = (int(time[:2]) * 60 * 60) + (int(time[3:5]) * 60) + int(time[6:8])
        if val > 1.5:
            cut = i
            time = np1_time[i].split(" ")[1]
            start_time = (int(time[:2]) * 60 * 60) + (int(time[3:5]) * 60) + int(time[6:8])
            break

    back_cut = np1.shape[0]
    for i, val in enumerate(np1):
        if i > np1.shape[0]//2 and val < 1.4:
            back_cut = i+50
            break

    np1 = np1[:back_cut]
    np1_time = np1_time[:back_cut]

    test_start = np.array(test_start) + (start_time - first_time_powermeter)

    tick_seconds = 180 #180 = 1 min
    increment = 100
    tick, tick_labels = [], []
    for i in range(np1.shape[0]):
        if i%tick_seconds == 0: # and (i//tick_seconds)%increment == 0:
            tick.append(i)
            tick_labels.append((i//tick_seconds))


    lines = []
    time_diffs = []
    for i, val in enumerate(np1):
        time = np1_time[i].split(" ")[1]
        time_diff = ((int(time[:2]) * 60 * 60) + (int(time[3:5]) * 60) + int(time[6:8])) - start_time
        if time_diff in test_start and time_diff not in time_diffs:
            time_diffs.append(time_diff)
            lines.append(i)

    # plt.figure()
    # plt.subplot(211)
    # plt.plot(np.arange(np1.shape[0]), np1)
    #
    #
    # for i, xc in enumerate(lines):
    #     c = (random.uniform(0, 1), random.uniform(0, 1), random.uniform(0, 1))
    #     plt.axvline(x=xc, color=c, label=line_labels[i])
    #
    # plt.xticks(tick, tick_labels)
    # plt.xlabel("Minutes")
    # plt.ylabel("Watts")
    # plt.legend(bbox_to_anchor=(1, 1), loc='upper left', ncol=2)
    # plt.show()
    # exit()
    return np1, lines, line_labels, tick, tick_labels



def getData(filepath):
    print("Getting: ", filepath)
    # filepath = sys.argv[-1] #'/power_cunsumption/time_stamped_results/numpy/try0/numpy_timestamp_0.out'
    with open(filepath) as fp:
        arr = []
        line = fp.readline()
        cnt = 1
        while line:
            line = fp.readline()
            cnt += 1
            arr.append(line)


    start_time = 0
    test_dicts = {}
    last_line = [""]
    first = True
    test_name, test_start, test_ends = [], [], []
    most_recent_test = []


    for i, line in enumerate(arr):
        line_arr = line.split(" ")
        if len(line_arr) > 4:
            path = line_arr[3].split("/")
            if 'tests' in path:
                most_recent_test = line_arr

                if path[0] != last_line[3].split("/")[0] and last_line[3].split("/")[0] in list(test_dicts.keys()):
                    time = (int(line_arr[2][:2]) * 60 * 60) + (int(line_arr[2][3:5]) * 60) + int(line_arr[2][6:]) - start_time
                    test_dicts[last_line[3].split("/")[0]]['end'] = time
                    test_ends.append(time)

                if path[0] not in list(test_dicts.keys()):
                    if first:
                        time = 0
                        start_time = (int(line_arr[2][:2]) * 60 * 60) + (int(line_arr[2][3:5]) * 60) + int(line_arr[2][6:])
                        start_times = {'h':int(line_arr[2][:2]), 'm':int(line_arr[2][3:5]), 's':int(line_arr[2][6:])}
                        first = False
                    else:
                        time = (int(line_arr[2][:2]) * 60 * 60) + (int(line_arr[2][3:5]) * 60) + int(line_arr[2][6:]) - start_time

                    test_dicts[path[0]] = {}
                    test_dicts[path[0]]['start'] = time
                    test_name.append(path[0])
                    test_start.append(time)

        last_line = line_arr


    time = (int(most_recent_test[2][:2]) * 60 * 60) + (int(most_recent_test[2][3:5]) * 60) + int(most_recent_test[2][6:]) - start_time
    test_dicts[test_name[-1]]['end'] = time
    test_ends.append(time)
    test_name.append(test_name[-1])


    test_start.append(test_ends[-1])
    test_start = test_start[1:]
    test_name = test_name[1:]




    np1, lines, line_labels, tick, tick_labels = plotExperiment(test_start, test_name, filepath)
    return [np1, lines, line_labels, tick, tick_labels]

data_arrs = []
for p in sys.argv:
    if p[:3] != 'par':
        data_arrs.append(getData(p))

subplots(data_arrs)

# print(data_arrs)
