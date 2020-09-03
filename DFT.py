import numpy as np
import sys
import pandas as pd
import matplotlib.pyplot as plt



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


    return power

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

# numpy_tests:
# try1: 9212 - 9362
# try2: 8868 - 9000
# try3:

f_start = 9472
f_end = 9604

func = file_data[0][f_start:f_end]

start = f_end
end = f_end + func.shape[0]
step = 1
min_d = 10
min_d_idx = 0

min_med = 1000
min_med_idx = 0

while end < file_data[0].shape[0]:
# while end < 12500:
    window = file_data[0][start:end]
    window_fft = np.absolute(np.fft.fft(window)[1:((end-start)//2)])

    target = func
    target_fft = np.absolute(np.fft.fft(target)[1:((end-start)//2)])

    window_fft_diff = np.absolute(np.fft.ifft(target_fft/window_fft))

    if np.median(window) > 1.5: #filter out times with nothing running

        # difference of medians
        diff_med = np.absolute(np.median(window) - np.median(target))
        # if diff_med < min_med:
        #     min_med = diff_med
        #     min_med_idx = start

        med_abs_val = np.median(np.absolute(window_fft_diff))
        argmax = np.argmax(window_fft_diff)
        h_p = np.amax(window_fft_diff)
        w_p, left, right = getWidth(window_fft_diff, med_abs_val)
        L_n = np.amax(np.concatenate((window_fft_diff[:left],window_fft_diff[right:]), axis=0)).real

        d = ((w_p * (L_n / h_p)).real) * diff_med
        if d == 0:
            print("d: ", d)
            print("median abs val: ", med_abs_val)
            print("argmax: ", argmax)
            print("width: ", w_p, " l: ", left, " r: ", right)
            print("outside of pulse: ", L_n)
            plt.plot(np.arange(window_fft_diff.shape[0]), window_fft_diff)
            plt.show()
            exit()



        if d < min_d:
            min_d_idx = start
            min_d = d
    # print("************************************")
    # print("d: ", d)
    # print("median abs val: ", med_abs_val)
    # print("argmax: ", argmax)
    # print("width: ", w_p, " l: ", left, " r: ", right)
    # print("outside of pulse: ", L_n)
    # plt.plot(np.arange(window_fft_diff.shape[0]), window_fft_diff)
    # plt.show()
    start+=step
    end+=step

print("Min d: ", min_d)
plt.axvline(x=min_d_idx, color='r')
# plt.axvline(x=min_med_idx, color='g')
plt.plot(np.arange(file_data[0].shape[0]), file_data[0])
plt.show()
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
