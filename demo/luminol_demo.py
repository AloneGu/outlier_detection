import numpy as np
from outlier_detector.base_detector import OutlierDetector

X = list(np.random.randn(20, 1).flatten()) + [1, 2, 23, 25, 5]

for algo in ['bitmap', 'derivative', 'exp_avg']:
    print algo
    detector = OutlierDetector(algo_name=algo)
    print detector.predict(X)
    print detector.score(X)
    print '================='