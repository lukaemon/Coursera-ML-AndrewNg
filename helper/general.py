import numpy as np


def get_X(df):
    return df.iloc[:, :-1].as_matrix()


def get_y(df):
    '''assume the last column is the target'''
    return np.array(df.iloc[:, -1])


def normalize_feature(df):
    return df.apply(lambda s: (s - s.mean()) / s.std())
