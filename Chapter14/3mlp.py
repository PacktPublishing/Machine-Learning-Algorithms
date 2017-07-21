from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.contrib.layers as tfl

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

from mpl_toolkits.mplot3d import Axes3D

nb_samples = 1000
nb_features = 3
nb_epochs = 200
batch_size = 50

# For reproducibility
np.random.seed(1000)


if __name__ == '__main__':
    # Create the dataset
    X, Y = make_classification(n_samples=nb_samples, n_features=nb_features,
                               n_informative=3, n_redundant=0, n_classes=2, n_clusters_per_class=3)

    # Show the dataset
    fig = plt.figure(figsize=(11, 11))
    ax = fig.add_subplot(111, projection='3d')

    for i, x in enumerate(X):
        if Y[i] == 0:
            ax.scatter(x[0], x[1], x[2], marker='s', color='blue')
        elif Y[i] == 1:
            ax.scatter(x[0], x[1], x[2], marker='d', color='red')

    ax.set_xlabel(r'$X_0$')
    ax.set_ylabel(r'$X_1$')
    ax.set_zlabel(r'$X_2$')
    plt.show()

    # Create train and test sets
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

    # Create the graph
    graph = tf.Graph()

    with graph.as_default():
        Xt = tf.placeholder(tf.float32, shape=(None, nb_features), name='X')
        Yt = tf.placeholder(tf.float32, shape=(None, 1), name='Y')

        layer_1 = tfl.fully_connected(Xt, num_outputs=50, activation_fn=tf.tanh)
        layer_2 = tfl.fully_connected(layer_1, num_outputs=1, activation_fn=tf.sigmoid)

        Yo = tf.round(layer_2)

        loss = tf.nn.l2_loss(layer_2 - Yt)
        training_step = tf.train.GradientDescentOptimizer(0.025).minimize(loss)

    session = tf.InteractiveSession(graph=graph)
    tf.global_variables_initializer().run()

    # Run the training cycle
    for e in range(nb_epochs):
        total_loss = 0.0
        Xb = np.ndarray(shape=(batch_size, nb_features), dtype=np.float32)
        Yb = np.ndarray(shape=(batch_size, 1), dtype=np.float32)

        for i in range(0, X_train.shape[0] - batch_size, batch_size):
            Xb[:, :] = X_train[i:i + batch_size, :]
            Yb[:, 0] = Y_train[i:i + batch_size]

            loss_value, _ = session.run([loss, training_step], feed_dict={Xt: Xb, Yt: Yb})
            total_loss += loss_value

        Y_predicted = session.run([Yo], feed_dict={Xt: X_test.reshape((X_test.shape[0], nb_features))})
        accuracy = 1.0 - (np.sum(np.abs(np.array(Y_predicted[0]).squeeze(axis=1) - Y_test)) / float(Y_test.shape[0]))

        print('Epoch %d) Total loss: %.2f - Accuracy: %.2f' % (e, total_loss, accuracy))
