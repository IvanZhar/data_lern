import numpy as np


class LinearRegression:
    def __init__(self, delta, dataset, y='y'):
        self.delta = delta
        self.data = dataset
        self.data['x-1'] = np.ones((len(dataset),))
        self.l = len(self.data)
        self.y = self.data[y].to_numpy().reshape(len(self.data), 1)
        self.x = self.data.iloc[:, 1:].to_numpy()

    def loss_function(self, w):
        return (1 / self.l) * (np.linalg.norm(np.dot(self.x, w) - self.y)) ** 2

    def gradient_calc(self, w):
        gradient = (2 / self.l) * np.dot(self.x.transpose(), np.dot(self.x, w) - self.y)
        return gradient

    def hessian_calc(self):
        hessian = (2 / self.l) * np.dot(self.x.transpose(), self.x)
        return hessian

    def step_calc(self, alpha, p, w):
        return (1 / self.l) * (np.linalg.norm(np.dot(self.x, w + (alpha * p)) - self.y)) ** 2

    def main(self, delta):
        w = np.ones((self.x[0].shape[0], 1))
        prec = False
        while not prec:
            grad, hessian = self.gradient_calc(w), self.hessian_calc()
            print(self.loss_function(w))
            if np.linalg.norm(grad) < delta:
                return w
            p = -1 * np.dot(hessian, grad)
            alpha = optimize.golden(self.step_calc, args=(p, w), full_output=True)[0]
            w = w + alpha * p