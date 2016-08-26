import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib

def get_X(df):
    """
    use concat to add intersect feature to avoid side effect
    not efficient for big dataset though
    """
    ones = pd.DataFrame({'ones': np.ones(len(df))})
    data = pd.concat([ones, df], axis=1)  # column concat
    return data.iloc[:, :-1].as_matrix()  # this return ndarray, not matrix


def get_y(df):
    '''assume the last column is the target'''
    return np.array(df.iloc[:, -1])


def normalize_feature(df):
    return df.apply(lambda s: (s - s.mean()) / s.std())


def plot_an_image(a_raw_im):
    """
    a_raw_im : (400, )
    """
    im = a_raw_im.reshape((20, 20))

    fig, ax = plt.subplots()
    ax.matshow(im, cmap=matplotlib.cm.binary)
    plt.xticks(np.array([]))  # just get rid of ticks
    plt.yticks(np.array([]))
