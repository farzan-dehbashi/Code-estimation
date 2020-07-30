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




file_data = []
for i in sys.argv[1:]:
    file_data.append(parseData(i))

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

# split = idx[diff > 200000]
# print(idx)
# print(diff.shape)
# exit()
# for i, xc in enumerate(split):
#     plt.axvline(x=xc, color='r')
# plt.plot(np.arange(all_power.shape[0]),all_power)
# plt.show()
# exit()
# large = np.where(diff > 200000)[0]

# for i, xc in enumerate(idx):
#     plt.axvline(x=xc, color='r')
# plt.plot(np.arange(all_power.shape[0]), all_power)
# plt.show()
# exit()

runs = []
for i in range(1,split.shape[0]):
    if split[i] - split[i-1] > 250000:
        runs.append(all_power[split[i-1]:split[i]])

for i,val in enumerate(runs):
    fname = 'power_cunsumption/2k_per_second_scipy/try'+str(i)+'/scipy_timestamped_try'+str(i)+'.npy'
    print("saving: ", fname)
    np.save(fname, val)
    # print(val.shape)
    # plt.plot(np.arange(val.shape[0]), val)
    # plt.show()
