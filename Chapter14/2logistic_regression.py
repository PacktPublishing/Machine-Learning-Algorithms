from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from sklearn.datasets import make_classification

# For reproducibility
np.random.seed(1000)

nb_samples = 500

if __name__ == '__main__':
    # Create the dataset
    X, Y = make_classification(n_samples=nb_samples, n_features=2, n_redundant=0, n_classes=2)

    # Plot the dataset
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.set_xlabel(r'$X_0$')
    ax.set_ylabel(r'$X_1$')

    for i, x in enumerate(X):
        if Y[i] == 0:
            ax.scatter(x[0], x[1], marker='d', color='blue')
        else:
            ax.scatter(x[0], x[1], marker='s', color='red')

    plt.show()

    # Create the graph
    graph = tf.Graph()

    with graph.as_default():
        Xt = tf.placeholder(tf.float32, shape=(None, 2), name='points')
        Yt = tf.placeholder(tf.float32, shape=(None, 1), name='classes')

        W = tf.Variable(tf.zeros((2, 1)), name='weights')
        bias = tf.Variable(tf.zeros((1, 1)), name='bias')

        Ye = tf.matmul(Xt, W) + bias
        Yc = tf.round(tf.sigmoid(Ye))

        loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=Ye, labels=Yt))
        training_step = tf.train.GradientDescentOptimizer(0.025).minimize(loss)

    session = tf.InteractiveSession(graph=graph)
    tf.global_variables_initializer().run()

    feed_dict = {
        Xt: X,
        Yt: Y.reshape((nb_samples, 1))
    }

    for i in range(10000):
        loss_value, _ = session.run([loss, training_step], feed_dict=feed_dict)
        if i % 100 == 0:
            print('Step %d, Loss: %.3f' % (i, loss_value))

    # Retrieve coefficients and intercept
    Wc, Wb = W.eval(), bias.eval()

    print('Coefficients:')
    print(Wc)

    print('Intercept:')
    print(Wb)

    # Plot the dataset with the separating hyperplane
    h = .02
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    Z = np.array(session.run([Yc], feed_dict={Xt: np.c_[xx.ravel(), yy.ravel()]}))

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.figure(1, figsize=(12, 12))
    plt.pcolormesh(xx, yy, Z, cmap=plt.cm.Pastel1)

    # Plot also the training points
    for i, x in enumerate(X):
        if Y[i] == 0:
            plt.scatter(x[0], x[1], marker='d', color='blue')
        else:
            plt.scatter(x[0], x[1], marker='s', color='red')

    plt.xlabel(r'$X_0$')
    plt.ylabel(r'$X_1$')

    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.xticks(())
    plt.yticks(())

    plt.show()