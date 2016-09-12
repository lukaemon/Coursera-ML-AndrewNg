import numpy as np
from scipy import stats
from sklearn.metrics import f1_score

# X data shape
# array([[ 13.04681517,  14.74115241],
#        [ 13.40852019,  13.7632696 ],
#        [ 14.19591481,  15.85318113],
#        [ 14.91470077,  16.17425987],
#        [ 13.57669961,  14.04284944]])


def select_threshold(X, Xval, yval):
    """use CV data to find the best epsilon
    Returns:
        e: best epsilon with the highest f-score
        f-score: such best f-score
        y_pred: the prediction of the best epsilon
    """
    # create multivariate model
    mu = X.mean(axis=0)
    cov = np.cov(X.T)
    multi_normal = stats.multivariate_normal(mu, cov)

    # this is key, use CV data for fine tuning hyper parameters
    pval = multi_normal.pdf(Xval)

    # set up epsilon candidates
    epsilon = np.linspace(np.min(pval), np.max(pval), num=10000)

    # calculate f-score
    fs = []
    for e in epsilon:
        y_pred = (pval <= e).astype('int')
        fs.append(f1_score(yval, y_pred))

    # find the best f-score
    argmax_fs = np.argmax(fs)

    return epsilon[argmax_fs], fs[argmax_fs], (multi_normal.pdf(Xval) <= epsilon[argmax_fs]).astype('int')
