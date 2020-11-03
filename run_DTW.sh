#!/bin/bash

# RQ1
# python3 DTW.py traces/numpy_random_funcs_nosleep_mean_SUBSET_long/numpy_random_funcs_nosleep_mean_SUBSET_long_try > rq1_mean.out
# python3 DTW.py traces/numpy_random_funcs_nosleep_median_SUBSET_long/numpy_random_funcs_nosleep_median_SUBSET_long_try > rq1_median.out
# python3 DTW.py traces/numpy_random_funcs_nosleep_fft_SUBSET/numpy_random_funcs_nosleep_fft_SUBSET_try > rq1_fft.out
# python3 DTW.py traces/numpy_random_funcs_nosleep_max_SUBSET_long/numpy_random_funcs_nosleep_max_SUBSET_long_try > rq1_amax.out

# RQ3
python3 DTW.py traces/numpy_random_funcs_nosleep_mean_SUBSET_long/numpy_random_funcs_nosleep_mean_SUBSET_long_try > rq3_mean.out
python3 DTW.py traces/numpy_random_funcs_nosleep_median_SUBSET_long/numpy_random_funcs_nosleep_median_SUBSET_long_try > rq3_median.out
python3 DTW.py traces/numpy_random_funcs_nosleep_fft_SUBSET/numpy_random_funcs_nosleep_fft_SUBSET_try > rq3_fft.out
python3 DTW.py traces/numpy_random_funcs_nosleep_max_SUBSET_long/numpy_random_funcs_nosleep_max_SUBSET_long_try > rq3_amax.out



# RQ4
# python3 DTW.py traces/noseed_numpy_random_funcs_mean/noseed_numpy_random_funcs_mean_try > rq4_mean.out
# python3 DTW.py traces/noseed_numpy_random_funcs_median/noseed_numpy_random_funcs_median_try > rq4_median.out
# python3 DTW.py traces/noseed_numpy_random_funcs_fft/noseed_numpy_random_funcs_fft_try > rq4_fft.out
# python3 DTW.py traces/noseed_numpy_random_funcs_max/noseed_numpy_random_funcs_max_try > rq4_amax.out
