"""
@Author Wu Wei
将逻辑回归看作是一个神经元的神经网络
"""

import numpy as np


class LogisticRegression:
    """
    逻辑回归算法，看作简单神经网络。

    Parameters
    ----------

        X : numpy.ndarray(nx, m).

        y : numpy.ndarray(1, m).

        m : int.
            The number of samples.

        nx : int
            The number of features.

        alpha : float, default 0.01.
            The learning rate of gradient descent algorithm.

        w : numpy.ndarray(nx, 1).
            The weight of this model.

        b : float, default 0.
            The bias value.

        cost : float.
            The cost value after training.

    """

    __X = None
    __y = None
    __m = 0
    __nx = 0
    __alpha = 0.01

    w = None
    b = 0.0
    cost = 0.0

    def __init__(self, X, y, alpha=0.01):
        self.__X, self.__y = X, y
        self.__nx, self.__m = self.__X.shape
        self.__alpha = alpha

    def set_X(self, X):
        self.__X = X

    def set_y(self, y):
        self.__y = y

    def set_alpha(self, alpha):
        self.__alpha = alpha

    @staticmethod
    def __serialize(w, b):
        return np.concatenate([w.ravel(), [b]])

    def __deserialize(self, param):
        w = param[:self.__nx].reshape(self.__nx, 1)
        b = param[-1]

        return w, b

    @staticmethod
    def sigmoid(z):
        return 1 / (1 + np.exp(-z))

    def forward_propagate(self, param):
        """
        前向传播算法。

        :param param: numpy.ndarray(nx+1,).
            Including w and b.
        :return: cost, a.
            The cost value and active value.
        """
        w, b = self.__deserialize(param)

        z = w.T @ self.__X + b
        a = self.sigmoid(z)
        lost = np.multiply(-self.__y, np.log(a)) - np.multiply(1 - self.__y, np.log(1 - a))
        cost = (1 / self.__m) * np.sum(lost)

        return cost, a

    def back_propagate(self, param):
        """
        反向传播算法。

        :param param: numpy.ndarray(nx+1,).
            Including w and b.
        :return: grad.
            The gradient of w and b.
        """
        _, a = self.forward_propagate(param)

        dw = (1 / self.__m) * np.dot(self.__X, (a - self.__y).T)
        db = (1 / self.__m) * np.sum(a - self.__y)

        grad = self.__serialize(dw, db)

        return grad

    def train(self, num_iteration=1000):
        """
        利用梯度下降算法，优化参数。

        :param num_iteration: 梯度下降算法的迭代次数。
        :return: self
        """

        costs = []

        param = np.zeros(self.__nx + 1)
        w, b = self.__deserialize(param)

        for i in range(num_iteration):
            cost = self.forward_propagate(param)
            costs.append(cost)

            grad = self.back_propagate(param)

            dw, db = self.__deserialize(grad)
            w = w - self.__alpha * dw
            b = b - self.__alpha * db

            param = self.__serialize(w, b)

        self.cost = self.forward_propagate(param)
        self.w, self.b = self.__deserialize(param)
        return self

    def get_predict(self):
        """
        计算训练样本的激活值。

        :return: numpy.ndarray(1, m).
            每一个样本的激活值。
        """
        param = self.__serialize(self.w, self.b)
        _, prob = self.forward_propagate(param)
        return (prob >= 0.5).astype(int)

    def get_accuracy(self):
        """
        计算训练样本的正确率。

        :return: float.
            准确率。
        """
        y_pred = self.get_predict().ravel()
        y = self.__y.ravel()
        correct = [1 if a == b else 0 for (a, b) in zip(y_pred, y)]
        accuracy = sum(map(int, correct)) / float(len(correct))
        return accuracy

    def predict(self, X):
        """
        计算样本X的激活值。

        :param X: numpy.ndarray(nx, m).
            输入样本。
        :return: numpy.ndarray(1, m).
            每一个样本的激活值。
        """

        z = self.w.T @ X + self.b
        prob = self.sigmoid(z)
        return (prob >= 0.5).astype(int)

    def accuracy(self, X, y):
        """
        计算样本X, y的准确率。

        :return: float.
            准确率。
        """
        y_pred = self.predict(X).ravel()
        y = y.ravel()
        correct = [1 if a == b else 0 for (a, b) in zip(y_pred, y)]
        accuracy = sum(map(int, correct)) / float(len(correct))
        return accuracy
