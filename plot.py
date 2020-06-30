import matplotlib.pyplot as plt
import numpy as np
from numpy import genfromtxt
import pandas as pd
import sys

# fname = sys.argv[-1]
# data = pd.read_csv(fname)
data = pd.read_csv(sys.argv[-1])
data2 = pd.read_csv(sys.argv[-2])
data3 = pd.read_csv(sys.argv[-3])



data = data[16:-10]
data = data.rename(columns={'Address': "time", 'USB0::0x2A8D::0x0101::MY59001058::0::INSTR': "power"})
data["power"] = pd.to_numeric(data["power"], downcast="float")
data2 = data2[16:-10]
data2 = data2.rename(columns={'Address': "time", 'USB0::0x2A8D::0x0101::MY59001058::0::INSTR': "power"})
data2["power"] = pd.to_numeric(data2["power"], downcast="float")
data3 = data3[16:-10]
data3 = data3.rename(columns={'Address': "time", 'USB0::0x2A8D::0x0101::MY59001058::0::INSTR': "power"})
data3["power"] = pd.to_numeric(data3["power"], downcast="float")

np1 = np.array(data['power'])*5
np2 = np.array(data2['power'])*5
np3 = np.array(data3['power'])*5

np1 = np1[:2500]
np2 = np2[:2500]
np3 = np3[:2500]

max_x = np.amax(np.array([np1.shape[0], np2.shape[0], np3.shape[0]]))
# x = np.arange(max_x)

arrs = [np1, np2, np3]

tick_seconds = 180
tick, tick_labels = [], []
for i in range(max_x):
    if i%tick_seconds == 0:
        tick.append(i)
        tick_labels.append((i//tick_seconds))	


plt.figure(figsize=(14,6))
for i, val in enumerate(arrs):
    plt.plot(np.arange(val.shape[0]), val, linewidth=1, alpha=0.5, label="run: " + str(i))
plt.ylabel('Watts', fontsize=20)
plt.xlabel('time (minutes)', fontsize=20)
plt.xticks(tick, tick_labels)
plt.title("Scipy Test Suite", fontsize=20)
plt.legend()
plt.show()

# dfs = [data, data2, data3]
#
# for df in dfs:
#     df = data.rename(columns={'Address': "time", 'USB0::0x2A8D::0x0101::MY59001058::0::INSTR': "power"})
#     df["power"] = pd.to_numeric(df["power"], downcast="float")
#
# print(data)
# exit()

# ax = plt.gca()
#
# data.plot(kind='line',x='time',y='power',ax=ax)
# data2.plot(kind='line',x='name',y='power', color='red', ax=ax)
# data3.plot(kind='line',x='name',y='power', color='red', ax=ax)
#
# plt.show()
