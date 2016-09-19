# create test data
import numpy as np

# fit and test
from outlier_detector.base_detector import OutlierDetector

x = list(np.arange(20)) + [10] * 100
t = OutlierDetector(algo_name='z_score')
t.fit(x)
print t.predict(x)
print t.score(x)
