# import numpy as np

# X data shape
# array([[ 13.04681517,  14.74115241],
#        [ 13.40852019,  13.7632696 ],
#        [ 14.19591481,  15.85318113],
#        [ 14.91470077,  16.17425987],
#        [ 13.57669961,  14.04284944]])


def estimate_Gaussian(X):
    """output mu and sigma2(variance) for every features in data set X
    Args:
        X (ndarray) (m, n)
    Returns:
        mu (ndarray) (n, )
        sigma2 (ndarray) (n, )
    """
    mu = X.mean(axis=0)
    sigma2 = X.var(axis=0)

    return mu, sigma2
