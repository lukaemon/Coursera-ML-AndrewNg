import numpy as np


def cost(theta, X, Y, R):
    """compute cost for every r(i, j)=1
    Args:
        theta (user, feature), (943, 10): user preference
        X (movie, feature), (1682, 10): movie features
        Y (movie, user), (1682, 943): (movie, user) rating
        R (movie, user), (1682, 943): (movie, user) has rating
    """
    inner = (X @ theta.T)[R == 1] - Y[R == 1]

    return np.dot(inner, inner) / 2
