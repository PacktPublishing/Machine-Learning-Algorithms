from __future__ import print_function

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from scipy.misc import face

# For reproducibility
np.random.seed(1000)

if __name__ == '__main__':
    # Load the image
    img = face(gray=True)

    # Show the original image
    plt.imshow(img, cmap='gray')
    plt.show()

    # Define the kernel
    kernel = np.array(
        [[0, 1, 0],
         [1, -4, 0],
         [0, 1, 0]],
        dtype=np.float32)

    cfilter = np.zeros((3, 3, 1, 1), dtype=np.float32)
    cfilter[:, :, 0, 0] = kernel

    # Create the graph
    graph = tf.Graph()

    with graph.as_default():
        x = tf.placeholder(tf.float32, shape=(None, 768, 1024, 1), name='image')
        f = tf.constant(cfilter)

        y = tf.nn.conv2d(x, f, strides=[1, 1, 1, 1], padding='SAME')

    session = tf.InteractiveSession(graph=graph)

    # Compute the convolution
    c_img = session.run([y], feed_dict={x: img.reshape((1, 768, 1024, 1))})
    n_img = np.array(c_img).reshape((768, 1024))

    # Show the final image
    plt.imshow(n_img, cmap='gray')
    plt.show()



