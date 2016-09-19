import numpy as np
from sklearn.svm import OneClassSVM


class SvmDetector(object):
    def __init__(self, param_dict={}):
        self.param_dict = param_dict
        print self.__class__.__name__, self.param_dict
        self.cls = OneClassSVM(**param_dict)

    def fit(self, x):
        self.cls.fit(x)

    def predict(self, x):
        res = self.cls.predict(x)
        # change to True / False , True means outlier
        return [True if r == 1 else False for r in res]

    def score(self, x):
        return np.abs(self.cls.decision_function(x)).flatten()
