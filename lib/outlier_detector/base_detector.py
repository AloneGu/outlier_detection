from svm_detector import SvmDetector
from bitmap_detector import BitmapDetector
from derivative_detector import DerivativeDetector
from exp_avg_detector import ExpAvgDetector
from modified_z_score_detector import ModifiedZScoreDetector
from hdbscan_detector import HdbscanDetector

anomaly_detector_algorithms = {
    'svm': SvmDetector,
    'bitmap': BitmapDetector,
    'derivative': DerivativeDetector,
    'exp_avg': ExpAvgDetector,
    'modified_z': ModifiedZScoreDetector,
    'hdbscan': HdbscanDetector
}


class OutlierDetector(object):
    """
    Base Class for AnomalyDetector algorithm.
    """

    def __init__(self, algo_name, param_dict={}):
        """
        Initializer
        :param str class_name: extended class name.

        """
        # self.algo = anomaly_detector_algorithms[algo_name]
        self.algo = anomaly_detector_algorithms[algo_name]
        self.model = self.algo(param_dict)

    def fit(self, x):
        self.model.fit(x)

    def predict(self, x):
        return self.model.predict(x)

    def score(self, x):
        return self.model.score(x)
