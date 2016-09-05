import numpy as np


class ZScoreDetector(object):
    def __init__(self, param_dict={}):
        self.thres = 3.5
        self.multi_param = 0.6745
        self.param_dict = {'thres': 3.5, 'multi_param': 0.6745}
        for k in ['thres', 'multi_param']:
            if k in param_dict:
                self.__setattr__(k, param_dict[k])
                self.param_dict[k] = param_dict[k]
        print self.__class__.__name__, self.param_dict
        self.mean = None
        self.mean_deviation = None

    def fit(self, x):
        x = self._trans_type(x)
        # get data median and med_abs_deviation
        if len(x.shape) == 1:
            points = x[:, None]
        else:
            points = x
        self.mean = np.mean(points, axis=0)
        diff = np.sum((points - self.mean) ** 2, axis=-1)
        diff = np.sqrt(diff)
        self.mean_deviation = np.mean(diff)  # not be zero

    def predict(self, x):
        if self.mean is None:
            raise Exception('this model has no median value, please call fit before')
        points = self._trans_type(x)
        if len(points.shape) == 1:
            points = points[:, None]
        else:
            points = points
        diff = np.sum((points - self.mean) ** 2, axis=-1)
        diff = np.sqrt(diff)
        z_score = self.multi_param * diff / self.mean_deviation
        return z_score > self.thres

    def score(self, x):
        if self.mean is None:
            raise Exception('this model has no median value, please call fit before')
        points = self._trans_type(x)
        if len(points.shape) == 1:
            points = points[:, None]
        else:
            points = points
        diff = np.sum((points - self.mean) ** 2, axis=-1)
        diff = np.sqrt(diff)
        z_score = self.multi_param * diff / self.mean_deviation
        return z_score - self.thres

    def _trans_type(self, x):
        return np.array(x)
