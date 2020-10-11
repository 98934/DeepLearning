"""
@Author Wu Wei
深度神经网络模型
"""
import numpy as np
import matplotlib.pyplot as plt


class NeuralNetwork:
    """
    深度神经网络

    Parameter
    ---------
        layer : int, default 3, including, hidden layer and output layer, but excluding input layer.
            The layer's number of network layer.

        n : numpy.ndarray(layer,)[int].
            The n[l] is the unit's number of l-th layer.

        weight_shapes : numpy.ndarray(layer - 1,)[tuple].
            The weight_shapes[i] is the shape of i-th W.

        weight_size : int.
            The weight_size is the size of total W.

        __X : numpy.ndarray(nx, m).
            Input data.

        __Y : numpy.ndarray(1, m).
            Output data.

        __m : int.
            The number of samples.

        alpha : float, default 1.0.
            The alpha is the parameter of gradient descent algorithm.

        W : numpy.ndarray(layer + 1,)[numpy.ndarray].
            It's the weights after training.

        b : numpy.ndarray(layer + 1,)[numpy.ndarray].
            It's the bias after training.

    """
    layer = 3
    n = None
    __m = 0
    __X = None
    __Y = None
    alpha = 0.01

    weight_shapes = None
    weight_size = 0

    W = None
    b = None

    def __init__(self, X, Y, layer=3, n_h=(5, 5, 1), alpha=0.03):

        self.__X, self.__Y = X, Y
        self.__m = self.__X.shape[1]
        self.layer = layer
        self.alpha = alpha

        self.n = np.empty(self.layer + 1, dtype=int)
        self.n[0] = self.__X.shape[0]
        self.n[-1] = self.__Y.shape[0]
        for (k, u) in zip(range(1, self.layer + 1), n_h):
            self.n[k] = u

        self.weight_shapes = np.empty(self.layer + 1, dtype=tuple)
        for i in range(1, self.weight_shapes.shape[0]):
            self.weight_shapes[i] = (self.n[i], self.n[i - 1])
            self.weight_size += self.weight_shapes[i][0] * self.weight_shapes[i][1]

        # self.W, self.b = self.random_init()

    @staticmethod
    def sigmoid(z):
        return 1 / (1 + np.exp(-z))

    def sigmoid_gradient(self, z):
        return np.multiply(self.sigmoid(z), 1 - self.sigmoid(z))

    @staticmethod
    def ReLU(z):
        """
        Implement the ReLU function.
        """
        return np.maximum(0, z)

    @staticmethod
    def ReLU_gradient(z):
        """
        Compute the gradient of ReLU function.

        :param z: any shape array.
        :return: the gradient of ReLU function for parameter z.
        """
        return (z > 0).astype(float)

    @staticmethod
    def tanh(z):
        return np.divide(np.exp(z) - np.exp(-z), np.exp(z) + np.exp(-z))

    def tanh_gradient(self, z):
        return 1 - np.power(self.tanh(z), 2)

    def forward_propagation(self, W, b):
        """
        前向传播算法

        :param W: numpy.ndarray(layer + 1,)[numpy.ndarray(n[l], m)]
            weight.
        :param b: numpy.ndarray(layer + 1,)[numpy.ndarray(n[l], 1)]
            bias.
        :return:
            A: numpy.ndarray(layer + 1,)[numpy.ndarray(n[l], m)], active values.
            Z: numpy.ndarray(layer + 1,)[numpy.ndarray(n[l], m)].
            H: numpy.ndarray(n[layer], m), the y_hat values.
        """

        A = np.empty(self.layer + 1, dtype=np.ndarray)
        Z = np.empty(self.layer + 1, dtype=np.ndarray)
        A[0] = self.__X
        for i in range(1, self.layer):
            Z[i] = W[i] @ A[i - 1] + b[i]
            A[i] = self.ReLU(Z[i])
        Z[self.layer] = W[self.layer] @ A[self.layer - 1] + b[self.layer]
        A[self.layer] = self.sigmoid(Z[self.layer])

        Y_hat = A[self.layer]

        return A, Z, Y_hat

    def back_propagation(self, W, b):
        """
        Implement the back propagation algorithm.

        :param W: numpy.ndarray(layer + 1,)[numpy.ndarray(n[l], m)]
            weight.
        :param b: numpy.ndarray(layer + 1,)[numpy.ndarray(n[l], 1)]
            bias.
        :return: The gradient of W and b, their shapes same the shapes of W and b.
        """
        A, Z, Y_hat = self.forward_propagation(W, b)

        dA = np.empty(self.layer + 1, dtype=np.ndarray)
        dZ = np.empty(self.layer + 1, dtype=np.ndarray)
        dW = np.empty(self.layer + 1, dtype=np.ndarray)
        db = np.empty(self.layer + 1, dtype=np.ndarray)

        dA[self.layer] = -(np.divide(self.__Y, A[self.layer]) - np.divide(1 - self.__Y, 1 - A[self.layer]))
        assert(dA[self.layer].shape == (self.n[self.layer], self.__m))
        dZ[self.layer] = np.multiply(dA[self.layer], self.sigmoid_gradient(Z[self.layer]))
        dW[self.layer] = (1 / self.__m) * (dZ[self.layer] @ A[self.layer - 1].T)
        db[self.layer] = (1 / self.__m) * (np.sum(dZ[self.layer], axis=1, keepdims=True))
        dA[self.layer - 1] = W[self.layer].T @ dZ[self.layer]

        for i in range(self.layer - 1, 0, -1):
            dZ[i] = np.multiply(dA[i], self.ReLU_gradient(Z[i]))
            dW[i] = (1 / self.__m) * (dZ[i] @ A[i - 1].T)
            db[i] = (1 / self.__m) * np.sum(dZ[i], axis=1, keepdims=True)
            dA[i - 1] = W[i].T @ dZ[i]

        return dW, db

    def compute_cost(self, W, b):
        """
        The cost function of this neural network.

        :param W: numpy.ndarray(layer + 1,)[numpy.ndarray(n[l], m)]
            weight.
        :param b: numpy.ndarray(layer + 1,)[numpy.ndarray(n[l], 1)]
            bias.

        :return: float, the cost function value.
        """
        _, _, Y_hat = self.forward_propagation(W, b)
        first = -np.multiply(self.__Y, np.log(Y_hat))
        second = np.multiply((1 - self.__Y), np.log(1 - Y_hat))
        cost = (first - second).sum() / self.__m

        return cost

    def random_init(self):
        """
        Random initialize the W and b.

        :return: W, b.
        """
        W = np.empty(self.layer + 1, dtype=np.ndarray)
        b = np.empty(self.layer + 1, dtype=np.ndarray)
        for i in range(1, W.shape[0]):
            W[i] = np.random.uniform(-0.12, 0.12, self.weight_shapes[i])
        for i in range(1, W.shape[0]):
            b[i] = np.random.uniform(-0.12, 0.12, (self.n[i], 1))
        return W, b

    def train(self, num_iteration=5000):
        """
        Train the neural network by gradient descent algorithm.

        :return: self.
        """
        W, b = self.random_init()
        for i in range(1, W.shape[0]):
            print(f'W[{i}].shape = {W[i].shape}')
        for i in range(1, b.shape[0]):
            print(f'b[{i}].shape = {b[i].shape}')
        costs = [self.compute_cost(W, b)]

        for i in range(num_iteration):
            dW, db = self.back_propagation(W, b)

            for j in range(1, dW.shape[0]):
                W[j] = W[j] - self.alpha * dW[j]

            for j in range(1, db.shape[0]):
                b[j] = b[j] - self.alpha * db[j]

            cost = self.compute_cost(W, b)
            if i % 100 == 0:
                print(f'cost[{i}] = {cost}')
            costs.append(cost)

        x_axis = np.arange(len(costs))
        plt.plot(x_axis, costs)

        self.W, self.b = W, b
        return self

    def get_predict(self):
        """
        计算训练样本的预测值。

        :return: numpy.ndarray(1, m).
            每一个样本的预测值。
        """
        _, _, prob = self.forward_propagation(self.W, self.b)
        return (prob >= 0.5).astype(int)

    def get_accuracy(self):
        """
        计算训练样本的正确率。

        :return: float.
            准确率。
        """
        y_pred = self.get_predict().ravel()
        y = self.__Y.ravel()
        print(y_pred)
        print(y)
        correct = [1 if a == b else 0 for (a, b) in zip(y_pred, y)]
        accuracy = sum(map(int, correct)) / float(len(correct))
        return accuracy

    def predict(self, X):
        """
        计算X的预测值。

        :param X: numpy.ndarray(nx, m).
        :return: numpy.ndarray(1, m).
            每一个X样本的预测值。
        """
        A = np.empty(self.layer + 1, dtype=np.ndarray)
        Z = np.empty(self.layer + 1, dtype=np.ndarray)
        A[0] = X
        for i in range(1, self.layer):
            Z[i] = self.W[i] @ A[i - 1] + self.b[i]
            A[i] = self.ReLU(Z[i])
        Z[self.layer] = self.W[self.layer] @ A[self.layer - 1] + self.b[self.layer]
        A[self.layer] = self.sigmoid(Z[self.layer])

        Y_hat = A[self.layer]

        return (Y_hat >= 0.5).astype(int)

    def accuracy(self, X, Y):
        """
        计算测试样本X, Y的正确率。

        :return: float.
            准确率。
        """
        y_pred = self.predict(X).ravel()
        y = Y.ravel()
        print(y_pred)
        print(y)
        correct = [1 if a == b else 0 for (a, b) in zip(y_pred, y)]
        accuracy = sum(map(int, correct)) / float(len(correct))
        return accuracy
