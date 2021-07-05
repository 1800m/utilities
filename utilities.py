import numpy as np

class Utilities:
    def mean_squeared_error(self, y, t):
        return 0.5 * np.sum((y-t)**2)

