import numpy as np
import pandas as pd


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
    return np.array(df.iloc[:, -1]).reshape(len(df), 1)  # explicit shape for tensorflow


def normalize_feature(df):
    return df.apply(lambda s: (s - s.mean()) / s.std())
