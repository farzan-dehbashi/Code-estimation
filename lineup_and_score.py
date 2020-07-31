import numpy as np
import sys
from os import listdir
from matplotlib import pyplot as plt

og_path = "power_cunsumption/2k_per_second_"

test_idx = int(sys.argv[-2])

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

    test_start = np.array(test_start)*950
    return test_start, test_name


def getTestingSuite(lang):
    try_paths = [og_path + lang +"/"+f+"/" + lang +"_timestamped_"+f+".npy" for f in listdir(og_path + lang +"/") if f[:3] == "try"]
    traces, trace_test_times, trace_test_names = [], [], []

    # for count, path in enumerate(try_paths):
    #     print(">", path)
    #     if count > 1: break
    #     start = 0
    #     end = start+200
    #     step = 100
    #     arr = np.load(path)
    #     mean_arr = []
    #     while end < arr.shape[0]:
    #         mean_arr.append(np.mean(arr[start:end]))
    #         start += step
    #         end += step
    #     traces.append(np.array(mean_arr))
    for count, path in enumerate(try_paths):
        # print(path)
        log_fname = path[:-3] + "out"
        test_times, test_names = getData(log_fname, lang)
        trace_test_times.append(test_times)
        trace_test_names.append(test_names)

        # if count > 2: break
        arr = np.load(path)
        mn = []
        for i in range(1,arr.shape[0]):
            if i < 500:
                mn.append(np.mean(arr[:i]))
            else:
                mn.append(np.mean(arr[(i-500):i]))
        traces.append(np.array(mn))
        # plt.plot(np.arange(np.array(traces[-1]).shape[0]), np.array(traces[-1]))
        # for i,val in enumerate(test_times):
        #     plt.axvline(x=val, color='r')
        # plt.show()
        # exit()

    min_length = traces[0].shape[0]
    for i, val in enumerate(traces):
        if val.shape[0] < min_length:
            min_length = val.shape[0]
    cut_traces = []
    for i, val in enumerate(traces):
        cut_traces.append(val[:min_length])
    traces = np.array(cut_traces)


    # ------ set single test
    single_test_full = traces[-1]
    single_test_times = trace_test_times[-1]
    single_test_names = trace_test_names[-1]

    test_start = single_test_times[test_idx]
    test_end = single_test_times[test_idx+1]
    # test_start = np.where(single_test_names == 'fft')[0]
    single_test = single_test_full[test_start:test_end]

    # --- strip last test
    traces = traces[:-1]
    trace_test_times = trace_test_times[:-1]
    trace_test_names = trace_test_names[:-1]

    # mean of stats for previous tests
    traces = np.mean(traces,axis=0)
    ground_test_times = np.mean(np.array(trace_test_times),axis=0)
    # ------ single test

    return traces, single_test, ground_test_times, single_test_names

def getTrace(lang,testing_trace):
    return testing_trace[5000:6000]
    # return np.load(og_path+lang+"/single_test/"+sys.argv[-2]+".npy")


def densityPlot(error, trace, testing_trace, ground_test_times, single_test_names):
    normed_error = ((error - np.amin(error)) / (np.amax(error) - np.amin(error)))
    # print(np.amax(normed_error))
    # print(np.amin(normed_error))
    # print(normed_error)
    first = np.sort(normed_error)[3]
    normed_error_idx = np.where(normed_error < first)[0]
    normed_error = normed_error[normed_error_idx]


    normed_error = 1 - normed_error

    # colors = ['r']
    # for i,val in enumerate(normed_error):
    #     # print(val)
    #     plt.axvline(x=normed_error_idx[i], color='r',alpha=val)
    #     plt.axvline(x=(normed_error_idx[i]+trace.shape[0]), color='r',alpha=val)
    #
    # for i, val in enumerate(normed_error_idx):
    #     diff = np.absolute(val - ground_test_times[test_idx])
    #     print("error in seconds: ", diff / 950, " - percent of trace: ", diff / testing_trace.shape[0])
    #
    # for i, val in enumerate(ground_test_times):
    #     plt.axvline(x=val, color='c',alpha=0.5)

    # plt.plot(np.arange(testing_trace.shape[0]), testing_trace)
    # plt.show()
    # exit()
    # import matplotlib.pyplot as plt
    # import numpy as np; np.random.seed(1)
    plt.rcParams["figure.figsize"] = 5,2
    fig, (ax,ax2) = plt.subplots(nrows=2, sharex=True)

    # extent = [x[0]-(x[1]-x[0])/2., x[-1]+(x[1]-x[0])/2.,0,1]
    ax.imshow(error[np.newaxis,:], cmap="plasma", aspect="auto") #, extent=extent
    for i,val in enumerate(normed_error):
        ax.axvline(x=normed_error_idx[i], color='k',alpha=val)
        ax.axvline(x=(normed_error_idx[i]+trace.shape[0]), color='k',alpha=val)
    ax.axvline(x=ground_test_times[test_idx], color='c',alpha=val)
    ax.set_yticks([])
    # ax.set_xlim(extent[0], extent[1])

    ax2.plot(np.arange(testing_trace.shape[0]),testing_trace)

    plt.tight_layout()
    plt.show()
    exit()



def subplots(error, trace, testing_trace):
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1)
    fig.subplots_adjust(hspace=0.5)

    dt = 0.01
    t = np.arange(0, 30, dt)


    ax1.set_title('Numpy')
    # ax1.axvline(x=np.argmin(error), color='r')
    # ax1.axvline(x=(np.argmin(error)+trace.shape[0]), color='r')
    normed_error = (error - np.amin(error)) / (np.amax(error) - np.amin(error))

    for i,val in enumerate(normed_error):
        ax1.axvline(x=i, color='r',alpha=val)
    ax1.plot(np.arange(testing_trace.shape[0]), testing_trace)
    ax1.set_ylim(np.amin(testing_trace), np.amax(testing_trace))


    # plt.xticks(tick, tick_labels)
    # plt.xlabel("Minutes")
    # plt.ylabel("Watts")
    # ax1.legend(bbox_to_anchor=(1, 1), loc='upper left', ncol=2)


    # ax1.set_xticks(time_ticks)
    # ax1.set_xticklabels(time_labels)
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Watts')
    # ax1.grid(True)

    # cxy, f = ax2.csd(s1, s2, 256, 1. / dt)
    low_error = testing_trace[np.argmin(error):np.argmin(error)+trace.shape[0]]
    ax2.plot(np.arange(low_error.shape[0]), low_error, label='Tests')
    ax2.plot(np.arange(trace.shape[0]), trace, label='Trace')
    ax2.set_ylim(np.amin([np.amin(trace), np.amin(low_error)]), np.amax([np.amax(trace), np.amax(low_error)]))

    ax2.legend(bbox_to_anchor=(1, 1), loc='upper left', ncol=2)
    # ax2.set_xticks(time_ticks)
    # ax2.set_xticklabels(time_labels)
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Watts')

    ax3.plot(np.arange(error.shape[0]), error)
    ax3.set_ylim(np.amin(error), np.amax(error))
    ax3.axvline(x=np.argmin(error), color='r', label='min point')
    ax3.legend(bbox_to_anchor=(1, 1), loc='upper left', ncol=2)
    # ax3.set_xticks(time_ticks)
    # ax3.set_xticklabels(time_labels)
    ax3.set_xlabel('Time')
    ax3.set_ylabel('Error')

    plt.show()
    exit()

def calculateError(trace, testing_trace, ground_test_times, single_test_names):
    print("Calculating Error")
    error = []

    for i in range(0, testing_trace.shape[0]-trace.shape[0]):
        # if i <= trace.shape[0]: #beginning
        #     cat = trace[-i:]
        #     e = np.sum(np.absolute(np.subtract(testing_trace[:i], cat)))
        # elif (testing_trace.shape[0] - i) <= trace.shape[0]: #end
        #     cut = (testing_trace.shape[0] - i)
        #     cat = trace[:cut]
        #     e = np.sum(np.absolute(np.subtract(testing_trace[i:], cat)))
        # else:
            # e = np.sum(np.absolute(np.subtract(testing_trace[i:(i+trace.shape[0])], trace)))
        e = np.sum(np.absolute(np.subtract(testing_trace[i:(i+trace.shape[0])], trace))) / trace.shape[0]
        error.append(e)

    error = np.array(error)
    print("error: ", error.shape)
    print("max: ", np.amax(error))
    print("min: ", np.amin(error))
    densityPlot(error, trace, testing_trace, ground_test_times, single_test_names)
    # subplots(error, trace, testing_trace)
    #
    # plt.plot(np.arange(error.shape[0]), np.array(error))
    # plt.show()
    # exit()

if __name__ == "__main__":
    lang = sys.argv[-1]

    testing_trace, trace, ground_test_times, single_test_names = getTestingSuite(lang)
    # trace = getTrace(lang,arr)

    # plt.plot(np.arange(trace.shape[0]), trace)
    # plt.show()

    calculateError(trace, testing_trace, ground_test_times, single_test_names)
