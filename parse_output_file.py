import pandas as pd
import numpy as np
import sys
import random
import copy
from matplotlib import pyplot as plt
from matplotlib.font_manager import FontProperties


def subplots(data_arrs):

    min_len = np.amin(np.array([data_arrs[0][0].shape, data_arrs[1][0].shape, data_arrs[2][0].shape]))
    time_labels = []
    time_ticks = []
    minute = 180
    for i in range(min_len):
        if i%minute == 0:
            time_labels.append(str(i//minute))
            time_ticks.append(i)


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
    ax1.plot(np.arange(min_len), data_arrs[0][0][:min_len])
    ax1.set_ylim(min_y, max_y)

    data_arrs[0][3] = data_arrs[0][3][1:]
    for i, xc in enumerate(data_arrs[0][6]):
        ax1.axvline(x=xc, color=colors[i], label=str(data_arrs[0][3][i]))

    # plt.xticks(tick, tick_labels)
    # plt.xlabel("Minutes")
    # plt.ylabel("Watts")
    ax1.legend(bbox_to_anchor=(1, 1), loc='upper left', ncol=2)


    ax1.set_ylim(min_y, max_y)
    ax1.set_xticks(time_ticks)
    ax1.set_xticklabels(time_labels)
    ax1.set_xlabel('Time (minutes)')
    ax1.set_ylabel('Watts')
    # ax1.grid(True)

    # cxy, f = ax2.csd(s1, s2, 256, 1. / dt)
    ax2.plot(np.arange(min_len), data_arrs[1][0][:min_len])
    ax2.set_ylim(min_y, max_y)
    data_arrs[1][3] = data_arrs[1][3][1:]
    for i, xc in enumerate(data_arrs[1][6]):
        ax2.axvline(x=xc, color=colors[i], label=str(data_arrs[1][3][i]))

    ax2.legend(bbox_to_anchor=(1, 1), loc='upper left', ncol=2)
    ax2.set_xticks(time_ticks)
    ax2.set_xticklabels(time_labels)
    ax2.set_xlabel('Time (minutes)')
    ax2.set_ylabel('Watts')

    ax3.plot(np.arange(min_len), data_arrs[2][0][:min_len])
    ax3.set_ylim(min_y, max_y)
    data_arrs[2][3] = data_arrs[2][3][1:]
    for i, xc in enumerate(data_arrs[2][6]):
        ax3.axvline(x=xc, color=colors[i], label=str(data_arrs[2][3][i]))

    ax3.legend(bbox_to_anchor=(1, 1), loc='upper left', ncol=2)
    ax3.set_xticks(time_ticks)
    ax3.set_xticklabels(time_labels)
    ax3.set_xlabel('Time (minutes)')
    ax3.set_ylabel('Watts')

    plt.show()
    print("shown")
    exit()


def parseExperiment_npy(test_start, line_labels,path, paths,lang):
    path = path[:-4]
    power = np.load(path + '.npy')


    test_ticks = []
    cut = 0
    start_time = 0
    mn = power[0]
    for i, val in enumerate(power):
        if i > 0:
            if i > 500: mn = np.mean(power[(i-500):i])
            else: mn = np.mean(power[:i])
        if val > 1.4 and (val - mn) > 0.2:
            cut = i
            break

    # numpy test time
    if lang == 'numpy':
        test_complete_time = 16*60*1000
    elif lang == 'scipy':
        test_complete_time = 12*60*1000

    back_cut = power.shape[0]
    for i, val in enumerate(power):
        if i > test_complete_time:
            mn = np.mean(power[i:(i+500)])
            if val < 1.4:
                back_cut = i
                break

    # plt.plot(np.arange(power.shape[0]), power)
    # plt.axvline(x=cut, c='r')
    # plt.axvline(x=back_cut, c='r')
    # plt.show()
    # exit()

    # power = power[cut:back_cut]
    power_time = np.arange(power.shape[0])/1000 #2000 because this was sampled with 2k per-second
    # test_start, line_labels
    test_time = []

    return power, power_time, test_time


def parseExperiment(test_start, line_labels,path, paths):
    path = path[:-4]
    data = pd.read_csv(path + '.csv')
    data = data[16:-10]
    data = data.rename(columns={'Address': "time", 'USB0::0x2A8D::0x0101::MY59001058::0::INSTR': "power"})
    data["power"] = pd.to_numeric(data["power"], downcast="float")
    power = np.array(data['power'])*5
    power_time = np.array(data['time'])



    test_ticks = []
    cut = 0
    start_time = 0
    for i, val in enumerate(power):
        if i == 0:
            time = power_time[i].split(" ")[1]
            first_time_powermeter = (int(time[:2]) * 60 * 60) + (int(time[3:5]) * 60) + int(time[6:8])
        if val > 1.5:
            cut = i
            time = power_time[i].split(" ")[1]
            start_time = (int(time[:2]) * 60 * 60) + (int(time[3:5]) * 60) + int(time[6:8])
            cut_time = time[:-4]
            break

    # ----- this shows the start times for each run -----
    # plt.plot(np.arange(cut+50), power[:cut+50])
    # plt.axvline(x=cut, color='r')
    # plt.show()


    back_cut = power.shape[0]
    for i, val in enumerate(power):
        if i > power.shape[0]//2 and val < 1.4:
            back_cut = i #+50
            back_cut_time = power_time[i]
            break


    power = power[cut:back_cut]
    power_time = power_time[cut:back_cut]
    # test_start, line_labels
    test_time = []
    test_start = test_start[1:]
    for i, val in enumerate(power_time):
        time = val.split(" ")[1]
        power_time[i] = ((int(time[:2]) * 60 * 60) + (int(time[3:5]) * 60) + int(time[6:8])) - start_time
        if len(test_start) > 0:
            if power_time[i] == test_start[0]:
                test_time.append(i)
                test_start = test_start[1:]

    return power, power_time, test_time


def compare(timeslice, data, start, stop, tests=None):
    ''' This calculates mean and standard deviation along an axis for a given time segment

    in:
        timeslice: (3, [len of interest])
        data: (3, [len of experiment])
        tests: (3, [test strings of interest])
        start: start time within experiment
        stop: stop time of timeslice within experiment

    returns:
        similar_traces: (?, 60) traces that are within standard deviation
        perc: (?, ) Percentage of samples in trace that are within standard deviation
        # in_range: array of how many values of a timeslice are outside of standard deviation

    '''

    mean = np.mean(timeslice, axis=0) # 60,
    std = np.std(timeslice, axis=0) # 60,
    mean_p_std = mean + std
    mean_m_std = mean - std

    in_range = []
    for i, val in enumerate(timeslice):
        temp_val = copy.deepcopy(val)
        temp_val[(temp_val < mean_m_std) | (temp_val > mean_p_std)] = -1
        in_range.append(np.count_nonzero(temp_val > 0))

    # ----- this shows the segment of interest in a subplot showing the entire test sequence for reference -----
    # plotSubplotandReference(data, timeslice, start, stop, tests)

    tau = 0.0
    # print(np.mean(in_range))
    perc = np.array(in_range) / timeslice.shape[1]
    idx = np.argwhere(perc >= tau)
    similar_traces = np.squeeze(timeslice[idx], 1)
    return similar_traces, idx, perc #np.mean(in_range)

def plotSubplotandReference(data, slice, start, stop, tests):
    '''this shows the segment of interest in a subplot showing the entire test sequence for reference'''
    # print("Tests: ", tests)
    fig, (ax1, ax2) = plt.subplots(2, 1)
    fig.subplots_adjust(hspace=0.5)
    ax1.plot(np.arange(data.shape[1]), data[0])
    ax1.plot(np.arange(data.shape[1]), data[1])
    ax1.plot(np.arange(data.shape[1]), data[2])
    ax1.axvline(x=start, color='r')
    ax1.axvline(x=stop, color='r')
    ax2.plot(np.arange(slice.shape[1]), slice[0])
    ax2.plot(np.arange(slice.shape[1]), slice[1])
    ax2.plot(np.arange(slice.shape[1]), slice[2])
    plt.show()


def analyze_fromNumpyArray(data_arrs,lang):
    compare_same_length = False

    data = []
    min_length_power = 100000000000000
    for i, val in enumerate(data_arrs):
        # print(val[0].shape)
        if val[0].shape[0] < min_length_power:
            min_length_power = val[0].shape[0]

        # plt.plot(np.arange(val[0].shape[0]), val[0])
        # plt.show()

    # ----- all are about the same length, so compare all of them together
    if compare_same_length:
        data, all_power_times = [], []
        for i, val in enumerate(data_arrs):
            data.append(val[0][:min_length_power])
            all_power_times.append(val[1][:min_length_power])

        data = np.array(data)
        all_power_times = np.array(all_power_times)

        start = 0
        stop = start + 1000
        step = 500
        similar_trace_arr = []

        while stop < data.shape[1]:
            power_slice = data[:,start:stop]
            power_time_slice = all_power_times[:,start:stop]

            similar_traces, idx, trace_scores = compare(power_slice, data, start, stop)

            if similar_traces.size > 0:
                similar_trace_arr.append(similar_traces)
            start+=step
            stop+=step
            # if start == 800: #------ early stop
            #     exit()
        if len(similar_trace_arr) > 0:
            similar_traces = np.concatenate(similar_trace_arr, axis=0)
    else: # ----- take all samples
        similar_trace_arr = []
        for i, val in enumerate(data_arrs):
            start = 0
            # stop = start + 1000
            step = 200
            idx = []
            cur = val[0]
            while cur.shape[0] > 0:
                if cur.shape[0]%step == 0:
                    break
                else:
                    cur = cur[:-1]

            end_idx = step
            while end_idx < cur.shape[0]:
                idx.append(end_idx)
                end_idx+=step


            # similar_traces = np.array(np.split(cur,idx))

            similar_trace_arr.append(np.array(np.split(cur,idx)))

        similar_traces = np.concatenate(similar_trace_arr, axis=0)

    return similar_traces,[]

# data_arrs = power, power_time, test_start, test_name, paths, paths_times
def analyze(data_arrs):
    data = []
    min_length_power = 100000000000000
    for i, val in enumerate(data_arrs):
        if val[0].shape[0] < min_length_power:
            min_length_power = val[0].shape[0]


    # ------- pad the end with zeros
    all_power_times, all_paths, all_path_times = [], [], []
    for i, val in enumerate(data_arrs):
        # if val[0].shape[0] < max_length_power:
        #     power_vals = np.concatenate((np.array(val[0]), np.zeros((max_length_power - val[0].shape[0]))), axis=0)
        #     power_times = np.concatenate((np.array(val[1]), np.zeros((max_length_power - val[0].shape[0]))), axis=0)
        # else:
        #     power_vals = np.array(val[0])
        #     power_times = np.array(val[1])

        # -------- truncate to shortest length
        if val[0].shape[0] > min_length_power:
            power_vals = np.array(val[0])[:min_length_power]
            power_times = np.array(val[1])[:min_length_power]
        else:
            power_vals = np.array(val[0])
            power_times = np.array(val[1])

        data.append(power_vals)
        all_power_times.append(power_times)
        all_paths.append(np.array(val[4]))
        all_path_times.append(np.array(val[5]))

    data = np.array(data)
    all_power_times = np.array(all_power_times)
    all_paths = np.array(all_paths)
    all_path_times = np.array(all_path_times)


    # plt.plot(np.arange(min_length), data[0])
    # plt.plot(np.arange(min_length), data[1])
    # plt.plot(np.arange(min_length), data[2])
    # plt.show()
    # exit()

    start = data_arrs[2][-1][0] #first test
    stop = start + 60
    step = 60
    last_test_time = data_arrs[2][-1][-1]
    similar_test_arr, similar_trace_arr = [], []
    while stop < last_test_time:
        power_slice = data[:,start:stop]
        power_time_slice = all_power_times[:,start:stop]

#  ------------ actually prints tests for trace out

        last_test = {}
        for i, val in enumerate(power_time_slice):
            last_test[i] = "Setup"

        tests = []
        for i, val in enumerate(power_time_slice): # loop through 3 tests
            test_list = all_paths[i][(all_path_times[i] >= np.amin(val)) & (all_path_times[i] <= np.amax(val))]

            if test_list.size == 0:
                test_list = np.array(["Running"])
            tests.append(test_list)
            # tests.append(np.where(val))
            # print(last_test)
        tests = np.array(tests)
#  ------------ actually prints tests for trace out
        similar_traces, idx, trace_scores = compare(power_slice, data, start, stop, tests)

        similar_tests = np.squeeze(tests[idx], 1)
        if similar_traces.size > 0:
            similar_trace_arr.append(similar_traces)
            similar_test_arr.append(similar_tests)
        start+=step
        stop+=step
        # if start == 800: #------ early stop
        #     exit()

    if len(similar_trace_arr) > 0:
        similar_traces = np.concatenate(similar_trace_arr, axis=0)
        sim_test = []
        for i in similar_test_arr:
            for j in i:
                sim_test.append(j)
        similar_tests = np.array(sim_test)

    return similar_traces, similar_tests


def getData(filepath,lang):
    # print("Getting: ", filepath)
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
    paths, paths_times, all_timestamps = [], [], []
    first_test_time = 0

    for i, line in enumerate(arr):
        line_arr = line.split(" ")          # grabs line
        if len(line_arr) > 2: all_timestamps.append(line_arr[2]) # append test timestamp
        if i == 0:
            first_test_time = (int(line_arr[2][:2]) * 60 * 60) + (int(line_arr[2][3:5]) * 60) + int(line_arr[2][6:])

        if len(line_arr) > 4:               # is line long enough to be a test
            path = line_arr[3].split("/")
            if 'tests' in path:
                paths.append(line_arr[3])       # append path
                paths_times.append(((int(line_arr[2][:2]) * 60 * 60) + (int(line_arr[2][3:5]) * 60) + int(line_arr[2][6:])) - first_test_time) #line_arr[2]

                most_recent_test = line_arr


                # ----- if dir is not same as last line && last line is a test -----> start of new test
                # if path[0] != last_line[3].split("/")[0] and last_line[3].split("/")[0] in list(test_dicts.keys()):
                #     time = (int(line_arr[2][:2]) * 60 * 60) + (int(line_arr[2][3:5]) * 60) + int(line_arr[2][6:]) - first_test_time #start_time
                #     test_dicts[last_line[3].split("/")[0]]['end'] = time
                #     test_ends.append(time)

                # ----- if first line of dir -----
                if path[0] not in list(test_dicts.keys()):
                    # if first:
                    #     time = 0
                    #     start_time = (int(line_arr[2][:2]) * 60 * 60) + (int(line_arr[2][3:5]) * 60) + int(line_arr[2][6:])
                    #     start_times = {'h':int(line_arr[2][:2]), 'm':int(line_arr[2][3:5]), 's':int(line_arr[2][6:])}
                    #     first = False
                    # else:
                    time = (int(line_arr[2][:2]) * 60 * 60) + (int(line_arr[2][3:5]) * 60) + int(line_arr[2][6:]) - first_test_time #start_time
                    test_dicts[path[0]] = {}
                    test_dicts[path[0]]['start'] = time
                    test_name.append(path[0])
                    test_start.append(time)
            elif i > 500:
                time = (int(line_arr[2][:2]) * 60 * 60) + (int(line_arr[2][3:5]) * 60) + int(line_arr[2][6:]) - first_test_time #start_time
                test_start.append(time)
                break
        last_line = line_arr


    # time = (int(most_recent_test[2][:2]) * 60 * 60) + (int(most_recent_test[2][3:5]) * 60) + int(most_recent_test[2][6:]) - start_time
    # test_dicts[test_name[-1]]['end'] = time
    # test_ends.append(time)
    test_name.append(test_name[-1]+"-end")

    # ----- this assumes the first "test" is compiling everything -----
    # test_start = test_start[1:]
    # test_name = test_name[1:]

    # print(len(paths))
    # print(len(paths_times))
    # exit()
    # np1, lines, line_labels, tick, tick_labels, np1_time = parseExperiment(test_start, test_name, filepath, paths)
    # power, power_time, test_time = parseExperiment(test_start, test_name, filepath, paths) #when you have .csvs
    power, power_time, test_time = parseExperiment_npy(test_start, test_name, filepath, paths,lang) #when you have npy arrays


    return [power, power_time, test_start, test_name, paths, paths_times, test_time] #[np1, lines, line_labels, tick, tick_labels, np1_time]

def buildDataset(paths,lang=None):
    data_arrs = []
    for p in paths:
        print(p)
        data_arrs.append(getData(p,lang))

    # similar_traces, similar_tests = analyze(data_arrs)
    similar_traces, similar_tests = analyze_fromNumpyArray(data_arrs,lang)
    return similar_traces, similar_tests


if __name__ == "__main__":
    data_arrs = []
    for p in sys.argv[1:]:
        data_arrs.append(getData(p))


    # similar_traces, similar_tests = analyze(data_arrs)
    subplots(data_arrs)
    exit()
