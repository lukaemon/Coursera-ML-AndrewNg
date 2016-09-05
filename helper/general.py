import numpy as np
import pandas as pd
import altair as alt


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


def scatter_plot(x, y, x_name, y_name):
    df = pd.DataFrame({x_name: x, y_name: y})
    c = alt.Chart(df).mark_circle().encode(
        x=x_name,
        y=y_name)
    return c


def line_plot(x, y, x_name, y_name):
    df = pd.DataFrame({x_name: x, y_name: y})
    c = alt.Chart(df).mark_line().encode(
        x=x_name,
        y=y_name)
    return c
