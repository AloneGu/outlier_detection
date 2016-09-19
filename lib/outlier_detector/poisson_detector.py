import numpy as np
from scipy import stats


class PoissonDetector(object):
    def __init__(self, param_dict={}):
        self.cdf_thres = 0.9  # 0.9 confidence interval
        self.param_dict = {'cdf_thres': 0.9}
        for k in ['pdf_thres']:
            if k in param_dict:
                self.__setattr__(k, param_dict[k])
                self.param_dict[k] = param_dict[k]
        print self.__class__.__name__, self.param_dict
        self.mean = None
        self.var = None

    def fit(self, x):
        data = self._trans_type(x)
        # one dimension or multi dimension
        if len(data.shape) == 1:
            self.uni_dimension_flag = True
            self.mean = data.mean()
            self.var = data.var()
        else:
            raise Exception('only support process 1D array')

    def predict(self, x):
        tmp_cdf = stats.poisson.cdf(x, loc=self.mean, mu=self.var)
        res = [self._check_confidence(item) for item in tmp_cdf]
        return self._trans_type(res)

    def score(self, x):
        tmp_cdf = stats.poisson.cdf(x, loc=self.mean, mu=self.var)
        res = [self._get_score(item) for item in tmp_cdf]
        return self._trans_type(res)

    def _trans_type(self, x):
        return np.array(x)

    def _check_confidence(self, x):
        return x > self.cdf_thres or 1 - x > self.cdf_thres

    def _get_score(self, x):
        return 10 * min(abs(x - self.cdf_thres), (abs(1 - x - self.cdf_thres)))
