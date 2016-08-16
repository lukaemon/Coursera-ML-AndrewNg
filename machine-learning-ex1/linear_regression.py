import numpy as np


def get_X(df):
    return df.iloc[:, :-1].as_matrix()


def get_y(df):
    '''assume the last column is the target'''
    return np.array(df.iloc[:, -1])


def compute_cost(X, y, theta):
    """
    X: R(m*n), m records, n features
    y: R(m)
    theta : R(n), linear regression parameters
    """
    inner = X @ theta - y  # R(m*1)
    square_sum = inner.T @ inner  # 1*m @ m*1 = 1*1

    cost = square_sum / (2 * (len(X)))

    return cost


def batch_update_theta(X, y, theta, alpha):
    """ return whole batch updated parameters
    n*m @ (m*1 - (m*n @ n*1)) -> n*1
    where n = n features
    """
    inner = X.T @ (X @ theta - y)  # R(n*1)

    new_theta = theta - (alpha / len(X)) * inner  # n*1

    return new_theta  # return theta vector R(1*n)


def batch_gradient_decent(X, y, theta, alpha, epoch):
    """ return the parameter and cost
    epoch: how many pass to run through whole batch
    """
    cost = [compute_cost(X, y, theta)]
    _theta = theta  # don't want to mess up with original theta

    for i in range(epoch):
        _theta = batch_update_theta(X, y, _theta, alpha)
        cost.append(compute_cost(X, y, _theta))

    return _theta, cost


def normalize_feature(df):
    return df.apply(lambda s: (s - s.mean()) / s.std())
