import numpy as np
import matplotlib.pyplot as plt
import planar_utils


def start():
    X, Y = planar_utils.load_planar_dataset()
    print(X.shape, Y.shape)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(X[0, :], X[1, :], c=np.squeeze(Y), s=40, cmap=plt.cm.Spectral)
    plt.show()


if __name__ == '__main__':
    start()
