import matplotlib.pyplot as plt
import numpy as np
from numpy import genfromtxt
import pandas as pd
import sys



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
max_y = max([np.amax(np1), np.amax(np2), np.amax(np3)])
min_y = min([np.amin(np1), np.amin(np2), np.amin(np3)])

x = np.arange(max_x)

tick_seconds = 180
tick, tick_labels = [], []
for i in range(max_x):
    if i%tick_seconds == 0:
        tick.append(i)
        tick_labels.append((i//tick_seconds))


fig, (ax1, ax2, ax3) = plt.subplots(3, 1)
# make a little extra space between the subplots
fig.subplots_adjust(hspace=0.5)

dt = 0.01
t = np.arange(0, 30, dt)


ax1.set_title('Numpy')
ax1.plot(x, np1)
ax1.set_ylim(min_y, max_y)
ax1.set_xticks(tick)
ax1.set_xticklabels(tick_labels)
ax1.set_xlabel('time')
ax1.set_ylabel('Watts')
# ax1.grid(True)

# cxy, f = ax2.csd(s1, s2, 256, 1. / dt)
ax2.plot(x, np2)
ax2.set_ylim(min_y, max_y)
ax2.set_xticks(tick)
ax2.set_xticklabels(tick_labels)
ax2.set_xlabel('time')
ax2.set_ylabel('Watts')

# ax2.set_ylabel('CSD (db)')
ax3.plot(x, np3)
ax3.set_ylim(min_y, max_y)
ax3.set_xticks(tick)
ax3.set_xticklabels(tick_labels)
ax3.set_xlabel('time')
ax3.set_ylabel('Watts')
plt.show()
