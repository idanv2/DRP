# only numpy,
# np.array(list)
# np.save(..)
# np.load
import math
import pickle
from pathlib import Path

import numpy as np

from constants import Constants
from utils import f_a


def generate_data(k1_train, k2_train):
    ex = []
    ey = []
    hx_x = []
    hy_x = []
    hx_y = []
    hy_y = []
    for k1 in k1_train:
        for k2 in k2_train:
            c = math.pi * (np.sqrt(k1 ** 2 + k2 ** 2))

            for n in range(2, Constants.TIME_STEPS + 2):
                ex.append(f_a(c, n - 2, k1, k2)[0])
                ey.append(np.vstack((f_a(c, n - 1, k1, k2)[0], f_a(c, n, k1, k2)[0])))
                hx_x.append(f_a(c, n - 2, k1, k2)[1])
                hx_y.append(np.vstack((f_a(c, n - 1, k1, k2)[1], f_a(c, n, k1, k2)[1])))
                hy_x.append(f_a(c, n - 2, k1, k2)[2])
                hy_y.append(np.vstack((f_a(c, n - 1, k1, k2)[2], f_a(c, n, k1, k2)[2])))

    return np.hstack(ex), np.hstack(ey), np.hstack(hx_x), np.hstack(hx_y), np.hstack(hy_x), np.hstack(hy_y)


k1 = [1., 3.]
k2 = [1., 3.]
ex, ey, hx_x, hx_y, hy_x, hy_y = generate_data(k1, k2)
pickle.dump(ex.reshape((len(k2) * len(k2) * Constants.TIME_STEPS, Constants.N, Constants.N,1)), open("files/ex.pkl", "wb"))
pickle.dump(hx_x.reshape((len(k2) * len(k2) * Constants.TIME_STEPS, Constants.N-2, Constants.N-1,1)), open("files/hx_x.pkl", "wb"))
pickle.dump(hy_x.reshape((len(k2) * len(k2) * Constants.TIME_STEPS, Constants.N-1, Constants.N-2,1)), open("files/hy_x.pkl", "wb"))
pickle.dump(ey.reshape((len(k2) * len(k2) * Constants.TIME_STEPS, Constants.N*2, Constants.N,1)), open("files/ey.pkl", "wb"))
pickle.dump(hx_y.reshape((len(k2) * len(k2) * Constants.TIME_STEPS, (Constants.N-2)*2, Constants.N-1,1)), open("files/hx_y.pkl", "wb"))
pickle.dump(hy_y.reshape((len(k2) * len(k2) * Constants.TIME_STEPS, (Constants.N-1)*2, Constants.N-2,1)), open("files/hy_y.pkl", "wb"))

