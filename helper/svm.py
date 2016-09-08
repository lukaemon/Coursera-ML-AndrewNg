import pandas as pd
import scipy.io as sio
import numpy as np


# support functions
def load_data(path):
    mat = sio.loadmat(path)
    print(mat.keys())
    data = pd.DataFrame(mat.get('X'), columns=['X1', 'X2'])
    data['y'] = mat.get('y')

    return data


def gaussian_kernel(x1, x2, sigma):
    return np.exp(- np.power(x1 - x2, 2).sum() / (2 * (sigma ** 2)))
