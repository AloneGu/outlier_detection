## Outlier Detection

* lib for outlier detection.

## Installation

        git clone xxx

        cd xxx

        sudo python setup.py install

### Supported algos:

                anomaly_detector_algorithms = {
                    'svm': SvmDetector,
                    'bitmap_detector': BitmapDetector,
                    'derivative_detector': DerivativeDetector,
                    'exp_avg_detector': ExpAvgDetector,
                    'modified_z_detector':ModifiedZScoreDetector
                }


### SVM demo:

                from outlier_detector.base_detector import OutlierDetector
                import numpy as np
                x = list(np.arange(20)) + [10]*100
                t = OutlierDetector(algo_name='modified_z_detector')
                t.fit(x)
                print t.predict(x)  # True means outlier
                print t.score(x)    # outlier scores


