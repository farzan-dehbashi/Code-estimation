import numpy as np
import sys
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch



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

    ending = power.shape[0]

    # power = np.concatenate((power[12640:12750], power[35682:35992], power[67294:67940]),axis=0) # mean grow
    # power = np.concatenate((power[4910:6960], power[8487:9757]),axis=0) # numpy/scipy fft
    # power = np.concatenate((power[8390:8990], power[10555:11150]),axis=0) #flatten/reshape
    # power = power[15350:16900]
    return power#[8400:9000] #[:ending//2] --- [10500:11200]

def makePaperFigure(file_data):
    xtick_vals, xtick_labels = [],[]
    for i in range(file_data[0].shape[0]):
        if i%200 == 0:
            xtick_vals.append(i)
            xtick_labels.append(round(i/1000,1))

    plt.figure(figsize=(14,5))
    # plt.axvspan(83, 535, color='orange', alpha=0.2)
    # plt.axvspan(671, 1070, color='red', alpha=0.2)
    # plt.axvspan(540, 920, color='blue', alpha=0.2)

    plt.plot(np.arange(file_data[0].shape[0]), file_data[0], c='b')
    plt.title("Code Execution", fontsize=25)
    plt.ylabel("Watts",fontsize=25)
    plt.xlabel("Time (s)",fontsize=25)
    # plt.xticks(xtick_vals, xtick_labels,fontsize=22)
    plt.yticks(fontsize=22)

    #
    # legend_elements = [ Patch(facecolor='orange', alpha=0.2,label='Flatten'),
    #                     Patch(facecolor='red', alpha=0.2,label='Reshape')]
    # #                     Patch(facecolor='blue',label='Large')]
    # plt.legend(handles=legend_elements, framealpha=1, fontsize=18, loc='upper right')
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


file_data = []
for i in sys.argv[1:]:
    file_data.append(parseData(i))

# ----- to make a nice figure of the power trace (Paper)
makePaperFigure(file_data)

f_start = 16224
f_end = 16355

func = file_data[0][f_start:f_end]

start = f_end
end = f_end + func.shape[0]
step = 1
min_d = 10
min_d_idx = 0

min_med = 1000
min_med_idx = 0

d_s = []

while end < file_data[0].shape[0]:
# while end < 12500:
    if end%100000 == 0:
        print("End @: ", end)
    window = file_data[0][start:end]
    window_fft = np.absolute(np.fft.fft(window)[1:((end-start)//2)])

    target = func
    target_fft = np.absolute(np.fft.fft(target)[1:((end-start)//2)])

    window_fft_diff = np.absolute(np.fft.ifft(target_fft/window_fft))

    # if np.median(window) > 1.5: #filter out times with nothing running

    # difference of medians
    diff_med = np.absolute(np.median(window) - np.median(target))

    med_abs_val = np.median(np.absolute(window_fft_diff))
    argmax = np.argmax(window_fft_diff)
    h_p = np.amax(window_fft_diff)
    w_p, left, right = getWidth(window_fft_diff, med_abs_val)
    L_n = np.amax(np.concatenate((window_fft_diff[:left],window_fft_diff[right:]), axis=0)).real

    amp = np.absolute(np.median(window) - np.median(target))

    d = (((w_p * (L_n / h_p)).real) * diff_med) * amp

    d_s.append(d)

    if d < min_d:
        min_d_idx = start
        min_d = d

    start+=step
    end+=step

# print("Min d: ", min_d)
# plt.axvline(x=min_d_idx, color='r')
# plt.axvline(x=min_med_idx, color='g')
# plt.plot(np.arange(file_data[0].shape[0]), file_data[0])
# plt.show()
# exit()

d_s = np.sort(np.array(d_s))
bunch = d_s[:500]
# print("len: ", d_s.shape[0])
# print("first: ", d_s[:100])
print("min: ", np.amin(bunch))
print("max: ", np.amax(bunch))
print("avg: ", np.mean(bunch))
print("median: ", np.median(bunch))
exit()

powers = []
times = []
if len(file_data) > 1:
    for i in file_data:
        powers.append(i[0])
        # times.append(i[1])

    all_power = np.squeeze(np.concatenate((powers), axis=0))
    # all_times = np.squeeze(np.concatenate((times), axis=0))
else:
    all_power = file_data[0]
    # all_times = file_data[0][1]

idx = np.where(all_power < 1.15)[0]
diff = np.diff(idx)
diff = np.concatenate((np.array([0]), diff))
# diff = np.sort(diff)
split_idx = np.where(diff > 200000)[0]
split = idx[split_idx]
split_m1 = idx[split_idx-1]
split = np.unique(np.concatenate((split, split_m1)))


runs = []
for i in range(1,split.shape[0]):
    if split[i] - split[i-1] > 250000:
        runs.append(all_power[split[i-1]:split[i]])

for i,val in enumerate(runs):
    # fname = 'power_cunsumption/2k_per_second_scipy/try'+str(i)+'/scipy_timestamped_try'+str(i)+'.npy'
    # print("saving: ", fname)
    # np.save(fname, val)
    print(val.shape)
    plt.plot(np.arange(val.shape[0]), val)
    plt.show()
