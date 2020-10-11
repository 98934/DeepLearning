"""
@Author Wu Wei
带有一个隐藏层的平面数据分类

"""
import numpy as np
import scipy.optimize as opt
from sklearn.metrics import classification_report


class NeuralNetwork:
    """
    带有一个隐藏层的神经网络模型

    Parameter
    ---------
    layer : int, default 3, including input layer, hidden layer and output layer.
        The layer's number of network layer.

    unit : numpy.ndarray(layer,)[int].
        The unit[l] is the unit's number of l-th layer.

    units_hidden_layer : list(layer - 2), default [5].
        The units of every hidden layer.

    theta_shapes : numpy.ndarray(layer - 1,)[tuple].
        The theta_shapes[i] is the shape of i-th Theta.

    theta_size : int.
        The theta_size is the size of total Thetas.

    X : numpy.ndarray(m, n).
        Input data.

    Y : numpy.ndarray(m, k).
        Output data.

    result : scipy.optimize.minimize, method='TNC'
        The result is the result trained by self.train().

    learning_rate : float, default 1.0.
        The learning_rate is the parameter of regularized term.

    """
    layer = 3
    unit = None
    units_hidden_layer = [4]
    theta_shapes = None
    theta_size = 0
    X = None
    Y = None
    result = None
    learning_rate = 1.0

    active_method = 'ReLU'

    def __init__(self, X, Y, units_hidden_layer=4, learning_rate=1.0, active_method='ReLU'):

        self.units_hidden_layer = units_hidden_layer

        self.X, self.Y = X, Y
        self.layer = 3
        self.units_hidden_layer = units_hidden_layer
        self.learning_rate = learning_rate
        self.active_method = active_method

        self.unit = np.empty(self.layer, dtype=int)
        self.unit[0] = self.X.shape[0]
        self.unit[-1] = self.Y.shape[0]
        self.unit[1] = self.units_hidden_layer

        self.theta_shapes = np.empty(self.layer - 1, dtype=tuple)
        for i in range(self.theta_shapes.shape[0]):
            self.theta_shapes[i] = (self.unit[i + 1], self.unit[i])
            self.theta_size += self.theta_shapes[i][0] * self.theta_shapes[i][1]

    @staticmethod
    def serialize(Thetas):
        """
        Convert Thetas to a 1-D numpy.ndarray.

        :param Thetas: numpy.ndarray(layer - 1,)[numpy.ndarray].
        :return: numpy.ndarray(theta_size,).
        """
        return np.concatenate([t.ravel() for t in Thetas])

    def deserialize(self, thetas):
        """
        Convert serialized thetas to (layer-1) Theta, and reshape to ndarray(layer-1,).

        :param thetas: numpy.ndarray(theta_size,).
        :return: numpy.ndarray(layer - 1,)[numpy.ndarray].
        """
        Thetas = np.empty(self.theta_shapes.shape, dtype=np.ndarray)
        st, ed = 0, 0
        for i in range(Thetas.shape[0]):
            shape = self.theta_shapes[i]
            ed = ed + self.theta_shapes[i][0] * self.theta_shapes[i][1]
            Thetas[i] = thetas[st:ed].reshape(shape)
            st = ed

        return Thetas

    @staticmethod
    def sigmoid(z):
        return 1 / (1 + np.exp(-z))

    def sigmoid_gradient(self, z):
        return np.multiply(self.sigmoid(z), 1 - self.sigmoid(z))

    def feed_forward(self, thetas):
        """
        前向传播算法

        :param thetas: numpy.ndarray(theta_size,).
        :return:
            A: numpy.ndarray(layer - 1,)[numpy.ndarray(m, unit[l])], active values.
            Z: numpy.ndarray(layer - 1,)[numpy.ndarray(m, unit[l])].
            H: numpy.ndarray(m, unit[layer-1]), the hypothesis values.
        """
        Thetas = self.deserialize(thetas)
        A = np.empty(self.layer, dtype=np.ndarray)
        Z = np.empty(self.layer, dtype=np.ndarray)
        A[0] = np.insert(self.X, 0, 1, axis=1)
        for i in range(1, self.layer):
            Z[i] = A[i - 1] @ Thetas[i - 1].T
            A[i] = np.insert(self.sigmoid(Z[i]), 0, 1, axis=1)

        H = np.delete(A[-1], 0, axis=1)

        return A, Z, H

    def cost(self, thetas):
        """
        The cost function of this neural network.

        :param thetas: thetas: numpy.ndarray(theta_size,).
        :return: float, the cost function value.
        """
        m = self.X.shape[0]

        _, _, H = self.feed_forward(thetas)
        first = -np.multiply(self.Y, np.log(H))
        second = np.multiply((1 - self.Y), np.log(1 - H))
        c = (first - second).sum() / m

        r = 0
        Thetas = self.deserialize(thetas)
        for T in Thetas:
            r += np.power(T[:, 1:], 2).sum()
        r = (self.learning_rate / (2 * m)) * r

        return c + r

    def gradient(self, thetas):
        """
        The gradient of thetas computed by back propagation algorithm.

        :param thetas: numpy.ndarray(theta_size,).
        :return: numpy.ndarray(theta_size,), the serialized gradient.
        """
        A, Z, H = self.feed_forward(thetas)
        Thetas = self.deserialize(thetas)
        Deltas = np.empty(Thetas.shape[0], dtype=np.ndarray)
        for i in range(Deltas.shape[0]):
            Deltas[i] = np.zeros(Thetas[i].shape)

        m = self.X.shape[0]
        deltas = np.empty(self.layer, dtype=np.ndarray)
        for i in range(m):
            h = H[i, :].reshape((H.shape[1], 1))
            y = self.Y[i, :].reshape((self.Y.shape[1], 1))
            deltas[-1] = h - y

            # Compute delta of each layer, ranging from (layer - 2) to 1.
            for layer in range(self.layer - 2, 0, -1):
                z = np.insert(Z[layer][i, :], 0, 1).reshape((Z[layer].shape[1] + 1, 1))
                deltas[layer] = np.multiply(Thetas[layer].T @ deltas[layer + 1], self.sigmoid_gradient(z))
                deltas[layer] = np.delete(deltas[layer], 0, axis=0)
                # print('delta[{}].shape = {}'.format(layer, deltas[layer].shape))

            # Accumulate the gradient of each sample.
            for layer in range(self.layer - 1):
                a = A[layer][i, :].reshape((1, A[layer].shape[1]))
                Deltas[layer] = Deltas[layer] + deltas[layer + 1] @ a

        for layer in range(self.layer - 1):
            Deltas[layer] = Deltas[layer] / m

        # Add regularized terms.
        for layer in range(self.layer - 1):
            Thetas[layer][:, 0] = 0
            reg_term = (self.learning_rate / m) * Thetas[layer]
            Deltas[layer] = Deltas[layer] + reg_term

        return self.serialize(Deltas)

    def train(self):
        """
        Train the neural network by scipy.optimize.minimize with 'TNC' method.

        :return: self.
        """
        thetas = np.random.uniform(-0.12, 0.12, self.theta_size)
        self.result = opt.minimize(fun=self.cost,
                                   x0=thetas,
                                   method='TNC',
                                   jac=self.gradient,
                                   options={'maxiter': 400})
        return self

    def predict(self):
        """
        预测数据

        :return: numpy.ndarray(m, k), the value is 1 or zero.
        """
        thetas = self.result.x
        _, _, H = self.feed_forward(thetas)

        rows = np.arange(self.X.shape[0], dtype='int64')
        cols = np.argmax(H, axis=1)
        Y_predict = np.zeros(H.shape)
        Y_predict[rows, cols] = 1

        return Y_predict

    def report(self):
        """
        分类报告，可查看准确率等信息

        :return: str, the classification_report
        """
        Y_predict = self.predict()
        rep = classification_report(self.Y, Y_predict)

        return rep
