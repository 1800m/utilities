import numpy as np

class Utilities:
    def mean_squeared_error(self, y, t):
        return 0.5 * np.sum((y-t)**2)

    def cross_entropy_error(self, y, t):
        delta = 1e-7
        return -np.sum(t * np.log(y + delta))

    def numerical_diff(self, f, x):
        h = 1e-4    # 0.0001
        return (f(x+h) - f(x-h)) / (2*h)
