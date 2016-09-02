import numpy as np
from hdbscan import HDBSCAN


class HdbscanDetector(object):
    def __init__(self, param_dict={}):
        self.param_dict = param_dict
        print self.__class__.__name__, self.param_dict
        self.cls = HDBSCAN(**param_dict)
        self.main_data = None

    def fit(self, x):
        data = self._transtype(x)
        self.main_data = data
        self.cls.fit(data)

    def predict(self, x):
        if self.main_data is None:
            raise Exception('this model has no main data, please call fit before')
        res = []
        data = self._transtype(x)
        for item in data:
            tmp_main_data = np.vstack((self.main_data, item))
            self.cls.fit(tmp_main_data)
            tmp_res = True if self.cls.labels_[-1] == -1 else False
            res.append(tmp_res)
        return res

    def score(self, x):
        if self.main_data is None:
            raise Exception('this model has no main data, please call fit before')
        res = []
        data = self._transtype(x)
        for item in data:
            if item in self.main_data:
                idx = np.where(self.main_data == item)[0][0]  # stupid numpy index
                self.cls.fit(self.main_data)
                tmp_res = self.cls.outlier_scores_[idx]
            else:
                tmp_main_data = np.vstack((self.main_data, item))
                self.cls.fit(tmp_main_data)
                tmp_res = self.cls.outlier_scores_[-1]

            res.append(tmp_res)
        return res

    def _transtype(self, x):
        return np.array(x)
