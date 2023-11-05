import numpy as np


class LaplaceDistribution:

    def __init__(self, features):
        self.loc = np.median(features, axis=0)
        self.scale = np.mean(np.abs(features - self.loc), axis=0)

    def logpdf(self, values):
        return -np.log(2 * self.scale) - np.abs(values - self.loc) / self.scale

    def pdf(self, values):
        log_pdf = self.logpdf(values)
        return np.exp(log_pdf)