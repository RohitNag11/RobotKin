import numpy as np


def chain_mat_mul(mats):
    res = mats[0]
    for i in range(1, len(mats)):
        res = np.matmul(res, mats[i])
    return res
