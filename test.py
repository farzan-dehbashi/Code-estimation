import numpy as np
import sys
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


def calculateD(sig1, sig2):

    # window_fft = np.absolute(np.fft.fft(sig1)[1:(sig1.shape[0]//2)])
    #
    # target_fft = np.absolute(np.fft.fft(sig2)[1:(sig2.shape[0]//2)])
    #
    # window_fft_diff = np.absolute(np.fft.ifft(target_fft/window_fft))

    map1 = np.absolute(np.fft.fft(sig1)[1:(sig1.shape[0]//2)])
    map2 = np.absolute(np.fft.fft(sig2)[1:(sig2.shape[0]//2)])

    map_diff = np.absolute(np.fft.ifft(map1/map2))

    # plt.plot(np.arange(map1.shape[0]), map1)
    # plt.plot(np.arange(map2.shape[0]), map2)
    plt.plot(np.arange(map_diff.shape[0]), map_diff)
    plt.show()
    exit()

    med_abs_val = np.median(np.absolute(window_fft_diff))
    argmax = np.argmax(window_fft_diff)
    h_p = np.amax(window_fft_diff)
    w_p, left, right = getWidth(window_fft_diff, med_abs_val)
    L_n = np.amax(np.concatenate((window_fft_diff[:left],window_fft_diff[right:]), axis=0)).real

    # amp = np.absolute(np.median(window) - np.median(target))
    amp = np.mean(np.absolute(window-target))

    # d = (((w_p * (L_n / h_p)).real) * diff_med) * amp
    d = ((w_p * (L_n / h_p)).real) * amp

    return d



time = np.arange(0, 10, 0.1)
sin_amp = np.sin(time)
cos_amp = np.cos(time)

calculateD(sin_amp, cos_amp)

# plt.plot(time, sin_amp)
# plt.plot(time, cos_amp)
# plt.show()
