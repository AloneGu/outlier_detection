from luminol import anomaly_detector

ALGO_NAME = 'derivative_detector'


class DerivativeDetector(object):
    def __init__(self, param_dict={}):
        self.param_dict = param_dict
        print self.__class__.__name__, self.param_dict
        self.cls = None

    def fit(self, x):
        tmp_dict = {}
        for i in range(len(x)):
            tmp_dict[i] = x[i]
        self.cls = anomaly_detector.AnomalyDetector(tmp_dict, algorithm_name=ALGO_NAME,
                                                    algorithm_params=self.param_dict)

    def predict(self, x):
        # this algo only support check anomaly in place, in other words, you don't have to use fit() directly
        self.fit(x)
        tmp_res = [False] * len(x)  # init value
        anomalies = self.cls.get_anomalies()
        for anomaly in anomalies:
            tmp_s = int(anomaly.start_timestamp)
            tmp_e = int(anomaly.end_timestamp)
            for x in range(tmp_s, tmp_e + 1):
                tmp_res[x] = True  # update anomaly index
        return tmp_res

    def score(self, x):
        self.fit(x)
        score_res = self.cls.get_all_scores()
        anom_score = [value for (timestamp, value) in score_res.iteritems()]
        return anom_score
