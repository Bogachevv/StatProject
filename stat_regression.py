import numpy as np
import matplotlib.pyplot as plt


class RegressionStat:
    def __init__(self, y_true, y_pred):
        self.y_true = y_true
        self.y_pred = y_pred

    def orig_dist(self):
        plt.hist(self.y_true, label='Distribution of true values')
        plt.show()

    def pred_dist(self):
        plt.hist(self.y_pred, label='Distribution of predicted values')
        plt.show()


