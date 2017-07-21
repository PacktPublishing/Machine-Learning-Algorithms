from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler, RobustScaler

# For reproducibility
np.random.seed(1000)

if __name__ == '__main__':
    # Create a dummy dataset
    data = np.ndarray(shape=(100, 2))

    for i in range(100):
        data[i, 0] = 2.0 + np.random.normal(1.5, 3.0)
        data[i, 1] = 0.5 + np.random.normal(1.5, 3.0)

    # Show the original and the scaled dataset
    fig, ax = plt.subplots(1, 2, figsize=(14, 5))

    ax[0].scatter(data[:, 0], data[:, 1])
    ax[0].set_xlim([-10, 10])
    ax[0].set_ylim([-10, 10])
    ax[0].grid()
    ax[0].set_xlabel('X')
    ax[0].set_ylabel('Y')
    ax[0].set_title('Raw data')

    # Scale data
    ss = StandardScaler()
    scaled_data = ss.fit_transform(data)

    ax[1].scatter(scaled_data[:, 0], scaled_data[:, 1])
    ax[1].set_xlim([-10, 10])
    ax[1].set_ylim([-10, 10])
    ax[1].grid()
    ax[1].set_xlabel('X')
    ax[1].set_ylabel('Y')
    ax[1].set_title('Scaled data')

    plt.show()

    # Scale data using a Robust Scaler
    fig, ax = plt.subplots(2, 2, figsize=(8, 8))

    ax[0, 0].scatter(data[:, 0], data[:, 1])
    ax[0, 0].set_xlim([-10, 10])
    ax[0, 0].set_ylim([-10, 10])
    ax[0, 0].grid()
    ax[0, 0].set_xlabel('X')
    ax[0, 0].set_ylabel('Y')
    ax[0, 0].set_title('Raw data')

    rs = RobustScaler(quantile_range=(15, 85))
    scaled_data = rs.fit_transform(data)

    ax[0, 1].scatter(scaled_data[:, 0], scaled_data[:, 1])
    ax[0, 1].set_xlim([-10, 10])
    ax[0, 1].set_ylim([-10, 10])
    ax[0, 1].grid()
    ax[0, 1].set_xlabel('X')
    ax[0, 1].set_ylabel('Y')
    ax[0, 1].set_title('Scaled data (15% - 85%)')

    rs1 = RobustScaler(quantile_range=(25, 75))
    scaled_data1 = rs1.fit_transform(data)

    ax[1, 0].scatter(scaled_data1[:, 0], scaled_data1[:, 1])
    ax[1, 0].set_xlim([-10, 10])
    ax[1, 0].set_ylim([-10, 10])
    ax[1, 0].grid()
    ax[1, 0].set_xlabel('X')
    ax[1, 0].set_ylabel('Y')
    ax[1, 0].set_title('Scaled data (25% - 75%)')

    rs2 = RobustScaler(quantile_range=(30, 65))
    scaled_data2 = rs2.fit_transform(data)

    ax[1, 1].scatter(scaled_data2[:, 0], scaled_data2[:, 1])
    ax[1, 1].set_xlim([-10, 10])
    ax[1, 1].set_ylim([-10, 10])
    ax[1, 1].grid()
    ax[1, 1].set_xlabel('X')
    ax[1, 1].set_ylabel('Y')
    ax[1, 1].set_title('Scaled data (30% - 60%)')

    plt.show()

