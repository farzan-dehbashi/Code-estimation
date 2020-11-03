#!/bin/bash

python3 DTW.py traces/noseed_numpy_random_funcs_mean/noseed_numpy_random_funcs_mean_try > rq4_mean.out

python3 DTW.py traces/noseed_numpy_random_funcs_median/noseed_numpy_random_funcs_median_try > rq4_median.out

python3 DTW.py traces/noseed_numpy_random_funcs_fft/noseed_numpy_random_funcs_fft_try > rq4_fft.out

python3 DTW.py traces/noseed_numpy_random_funcs_max/noseed_numpy_random_funcs_max_try > rq4_amax.out
