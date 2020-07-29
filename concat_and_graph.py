import numpy as np
import sys
import pandas as pd
import matplotlib.pyplot as plt



def parseData(path):
    data = pd.read_csv(path + '.csv')
    power = np.array(data[data.columns[1:2]])*5
    power_time = np.array(data[data.columns[:1]])

    return [power, power_time]




file_data = []
for i in sys.argv[1:]:
    file_data.append(parseData(i))

powers = []
times = []
for i in file_data:
    powers.append(i[0])
    times.append(i[1])


all_power = np.squeeze(np.concatenate((powers), axis=0))
all_times = np.squeeze(np.concatenate((times), axis=0))

plt.plot(all_times[500000:501000], all_power[500000:501000])
plt.show()
