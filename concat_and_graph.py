import numpy as np
import sys
import pandas as pd
import matplotlib.pyplot as plt



def parseData(path):
    # data = pd.read_csv(path + '.csv')
    # power = np.array(data[data.columns[1:2]])*5
    # power_time = np.array(data[data.columns[:1]])
    path = path[:-4]
    data = pd.read_csv(path + '.csv')
    data = data[16:-10]
    data = data.rename(columns={'Address': "time", 'USB0::0x2A8D::0x0101::MY59001058::0::INSTR': "power"})
    data["power"] = pd.to_numeric(data["power"], downcast="float")
    power = np.array(data['power'], dtype=np.float32)*5
    power_time = np.array(data['time'], dtype=np.float32)

    return [power, power_time]




file_data = []
for i in sys.argv[1:]:
    file_data.append(parseData(i))

powers = []
times = []
if len(file_data) > 1:
    for i in file_data:
        powers.append(i[0])
        times.append(i[1])

    all_power = np.squeeze(np.concatenate((powers), axis=0))
    all_times = np.squeeze(np.concatenate((times), axis=0))
else:
    all_power = file_data[0][0]
    all_times = file_data[0][1]

plt.plot(np.arange(all_power.shape[0]), all_power)
plt.show()
