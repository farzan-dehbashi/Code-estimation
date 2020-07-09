import pandas as pd
import numpy as np
import sys
from matplotlib import pyplot as plt

filepath = 'npy_tests.out'
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


test_start.append(test_ends[-1])

plt.figure()
plt.hlines(1,1,test_start[-1])  # Draw a horizontal line
plt.eventplot(test_start, orientation='horizontal', colors='b')
plt.axis('off')
plt.show()
