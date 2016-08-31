import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import scipy.io as sio
import helper.logistic_regression as lr


def load_data(path, transpose=True):
    data = sio.loadmat(path)
    y = data.get('y')  # (5000,1)
    y = y.reshape(y.shape[0])  # make it back to column vector

    X = data.get('X')  # (5000,400)

    if transpose:
        # for this dataset, you need a transpose to get the orientation right
        X = np.array([im.reshape((20, 20)).T for im in X])

        # and I flat the image again to preserve the vector presentation
        X = np.array([im.reshape(400) for im in X])

    return X, y


def load_weight(path):
    data = sio.loadmat(path)
    return data['Theta1'], data['Theta2']


def plot_an_image(image):
    """
    image : (20, 20)
    """
    fig, ax = plt.subplots(figsize=(1, 1))
    ax.matshow(image.reshape((20, 20)), cmap=matplotlib.cm.binary)
    plt.xticks(np.array([]))  # just get rid of ticks
    plt.yticks(np.array([]))


def plot_100_image(X):
    """ sample 100 image and show them
    X : (5000, 20, 20)
    """
    # sample 100 image, reshape, reorg it
    sample_idx = np.random.choice(np.arange(X.shape[0]), 100)  # 100*400
    sample_images = X[sample_idx, :]

    fig, ax_array = plt.subplots(nrows=10, ncols=10, sharey=True, sharex=True, figsize=(8, 8))

    for r in range(10):
        for c in range(10):
            ax_array[r, c].matshow(sample_images[10 * r + c].reshape((20, 20)),
                                   cmap=matplotlib.cm.binary)
            plt.xticks(np.array([]))
            plt.yticks(np.array([]))


def expand_y(y):
    """expand 5000*1 into 5000*10
    where y=10 -> [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
    """
    res = []
    for i in y:
        y_array = np.zeros(10)
        y_array[i - 1] = 1

        res.append(y_array)

    return np.array(res)


def feed_forward(X, t1, t2):
    """apply to architecture 400+1 * 25+1 *10"""
    a2 = lr.sigmoid(X @ t1.T)  # 5000*25
    a2 = np.insert(a2, 0, np.ones(a2.shape[0]), axis=1)

    a3 = lr.sigmoid(a2 @ t2.T)  # 5000*10, all number are still prob in [0, 1]

    y_pred = np.argmax(a3, axis=1) + 1  # 5000*1, comply with 1 index of matlab

    return expand_y(y_pred)
