import numpy as np
from scipy import stats


class GaussianDetector(object):
    def __init__(self, param_dict={}):
        self.pdf_thres = 0.11
        self.param_dict = {'pdf_thres': 0.11}
        for k in ['pdf_thres']:
            if k in param_dict:
                self.__setattr__(k, param_dict[k])
                self.param_dict[k] = param_dict[k]
        print self.__class__.__name__, self.param_dict
        self.mean = None
        self.std_or_cov = None
        self.uni_dimension_flag = True

    def fit(self, x):
        data = self._trans_type(x)
        # one dimension or multi dimension
        if len(data.shape) == 1:
            self.uni_dimension_flag = True
            self.mean = data.mean()
            self.std_or_cov = data.std()
        else:
            self.mean = data.mean(axis=0)
            self.std_or_cov = np.cov(data.T, bias=True)
            self.uni_dimension_flag = False

    def predict(self, x):
        if self.uni_dimension_flag:
            tmp_pdf = stats.norm.pdf(x, loc=self.mean, scale=self.std_or_cov)
        else:
            tmp_pdf = stats.multivariate_normal.pdf(x, mean=self.mean, cov=self.std_or_cov)
        res = [self._check_confidence(item) for item in tmp_pdf]
        return self._trans_type(res)

    def score(self, x):

        if self.uni_dimension_flag:
            tmp_pdf = stats.norm.pdf(x, loc=self.mean, scale=self.std_or_cov)
        else:
            tmp_pdf = stats.multivariate_normal.pdf(x, mean=self.mean, cov=self.std_or_cov)
        res = [self._get_score(item) for item in tmp_pdf]
        return self._trans_type(res)

    def _trans_type(self, x):
        return np.array(x)

    def _check_confidence(self, x):
        return x <= self.pdf_thres

    def _get_score(self, x):
        return 10 * abs(x - self.pdf_thres)
