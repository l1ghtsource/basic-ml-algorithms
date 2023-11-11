import numpy as np


class LossAndDerivatives:
    @staticmethod
    def mse(X, Y, w):
        error = X.dot(w) - Y
        if len(error.shape) > 1:
            return np.mean(np.sum(error ** 2, axis=1))
        else:
            return np.mean(error ** 2)

    @staticmethod
    def mae(X, Y, w):
        error = X.dot(w) - Y
        if len(error.shape) > 1:
            return np.mean(np.sum(np.abs(error), axis=1))
        else:
            return np.mean(np.abs(error))

    @staticmethod
    def l2_reg(w):
        return np.sum(w ** 2)

    @staticmethod
    def l1_reg(w):
        return np.sum(np.abs(w))

    @staticmethod
    def no_reg(w):
        return 0.0

    @staticmethod
    def mse_derivative(X, Y, w):
        error = X.dot(w) - Y
        if len(error.shape) > 1:
            return (2 / (X.shape[0] * error.shape[1])) * X.T.dot(error)
        else:
            return (2 / X.shape[0]) * X.T.dot(error)

    @staticmethod
    def mae_derivative(X, Y, w):
        error = X.dot(w) - Y
        if len(error.shape) > 1:
            return (1 / (X.shape[0] * error.shape[1])) * X.T.dot(np.sign(error))
        else:
            return (1 / X.shape[0]) * X.T.dot(np.sign(error))

    @staticmethod
    def l2_reg_derivative(w):
        return 2 * w

    @staticmethod
    def l1_reg_derivative(w):
        return np.sign(w)

    @staticmethod
    def no_reg_derivative(w):
        return np.zeros_like(w)
