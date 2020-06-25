import matplotlib.pyplot as plt
import numpy as np
from numpy import genfromtxt
import pandas as pd
import sys

fname = sys.argv[-1]
data = pd.read_csv(fname)

data = data[16:]
data = data.rename(columns={'Address': "time", 'USB0::0x2A8D::0x0101::MY59001058::0::INSTR': "power"})
data["power"] = pd.to_numeric(data["power"], downcast="float")

ax = plt.gca()

data.plot(kind='line',x='time',y='power',ax=ax)
# df.plot(kind='line',x='name',y='num_pets', color='red', ax=ax)

plt.show()
