import tensorflow as tf
import numpy as np


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

    return new_theta  # return theta vector R(n)


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


def linear_regression(X_data, y_data, alpha, epoch, optimizer=tf.train.GradientDescentOptimizer):
    """tensorflow implementation"""
    # placeholder for graph input
    X = tf.placeholder(tf.float32, shape=X_data.shape)
    y = tf.placeholder(tf.float32, shape=y_data.shape)

    # construct the graph
    with tf.variable_scope('linear-regression'):
        W = tf.get_variable("weights",
                            (X_data.shape[1], 1),
                            initializer=tf.constant_initializer())  # n*1

        y_pred = tf.matmul(X, W)  # m*n @ n*1 -> m*1

        loss = 1 / (2 * len(X_data)) * tf.matmul((y_pred - y), (y_pred - y), transpose_a=True)  # (m*1).T @ m*1 = 1*1

    opt = optimizer(learning_rate=alpha)
    opt_operation = opt.minimize(loss)

    # run the session
    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        loss_data = []

        for i in range(epoch):
            _, loss_val, W_val = sess.run([opt_operation, loss, W], feed_dict={X: X_data, y: y_data})
            loss_data.append(loss_val[0, 0])  # because every loss_val is 1*1 ndarray

            if len(loss_data) > 1 and np.abs(loss_data[-1] - loss_data[-2]) < 10 ** -9:  # early break when it's converged
                # print('Converged at epoch {}'.format(i))
                break

    # clear the graph
    tf.reset_default_graph()
    return {'loss': loss_data, 'parameters': W_val}  # just want to return in row vector format
