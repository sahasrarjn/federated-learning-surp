import numpy as np

def get_num_bits_numpy(x):
    count = 1
    for i in range(len(x.shape)):
        count *= x.shape[i]
    return x.dtype.itemsize * 8 * count
