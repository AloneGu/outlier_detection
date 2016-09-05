# create test data
import numpy as np

x = np.random.normal(0, 1, 100)
noise = [16, 17, 18, 19, 10]

# fit and test
from outlier_detector.base_detector import OutlierDetector

detector = OutlierDetector(algo_name='gaussian', param_dict={'pdf_thres': 0.1})
detector.fit(x)
print detector.predict(x[-5:])
print detector.score(x[-5:])
print detector.predict(noise)
print detector.score(noise)
