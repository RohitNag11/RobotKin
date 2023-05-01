import numpy as np


def chain_mat_mul(mats):
    res = mats[0]
    for i in range(1, len(mats)):
        res = np.matmul(res, mats[i])
    return res


def round(num, dp, round_down=False):
    if round_down:
        return np.floor(num * 10 ** dp) / 10 ** dp
    return np.ceil(num * 10 ** dp) / 10 ** dp
