import numpy as np
import pandas as pd
import scipy.optimize as so


class LinearRegression:
    def __init__(self, method='BFGS', delta=0.001):
        self.method = method
        self.delta = delta

    @staticmethod
    def get_gradient(w, x, y):
        return (2 / x.shape[0]) * np.dot(x.transpose(), np.dot(x, w) - y)

    @staticmethod
    def get_hessian(x):
        return (2 / x.shape[0]) * np.dot(x.transpose(), x)

    @staticmethod
    def get_loss(w, x, y):
        return (1 / x.shape[0]) * np.linalg.norm(np.dot(x, w), y) ** 2

    def get_step(self):
        pass

    def fit_bfgs(self, x, y):
        w = np.zeros((x.shape[1] + 1, 1))
        x = np.hstack((x, np.ones((x.shape[0], 1))))
        while True:
            gradient = self.get_gradient(w, x, y)
            hessian = self.get_hessian(x)
            p = -np.dot(hessian, gradient)
            line_search = so.line_search(self.get_loss, self.get_gradient, w, p, args=(x, y))
            print(line_search)
            exit()


    def fit(self, x, y):
        if self.method == 'BFGS':
            return self.fit_bfgs(x, y)


test_df = pd.read_csv('test_data.csv')
lin = LinearRegression()
lin.fit(test_df[['R&D Spend', 'Administration', 'Marketing Spend']].to_numpy(), test_df['Profit'].to_numpy().reshape(-1, 1))


