from outlier_detector.base_detector import OutlierDetector
import numpy as np
x = list(np.arange(20)) + [10]*100
t = OutlierDetector(algo_name='modified_z')
t.fit(x)
print t.predict(x)
print t.score(x)