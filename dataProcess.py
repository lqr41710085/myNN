import numpy as np


def norm(x):
    x_ = x - np.mean(x)
    x_ /= np.var(x)
    return x_


def dataProcess(x, y, flatten=True, oneHot_num=False, normal_regularization=True, shuffle=True):

    if flatten:
        x_ = np.asarray([xx.flatten() for xx in x], dtype=np.float32)
    else:
        x_ = x
    if oneHot_num:
        y_ = np.asarray(np.identity(oneHot_num)[y.reshape(y.shape[0],)])
    else:
        y_ = y.reshape(y.shape[0], 1)
    if normal_regularization:
        x_ = norm(x_)
    if shuffle:
        shuffle_index = np.random.permutation(np.arange(len(x_)))
        x_ = x_[shuffle_index]
        y_ = y_[shuffle_index]
    return x_, y_
