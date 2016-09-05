## Outlier Detection

* lib for outlier detection.

## Installation

        pip install git+ssh://git@github.com/AloneGu/outlier_detection.git

### Supported algos:

                anomaly_detector_algorithms = {
                    'svm': SvmDetector,
                    'bitmap': BitmapDetector,
                    'derivative': DerivativeDetector,
                    'exp_avg': ExpAvgDetector,
                    'modified_z': ModifiedZScoreDetector,
                    'z_score':ZScoreDetector,
                    'hdbscan': HdbscanDetector,
                    'gaussian':GaussianDetector,
                    'poisson':PoissonDetector
                }


### SVM demo:

                from outlier_detector.base_detector import OutlierDetector
                import numpy as np
                x = list(np.arange(20)) + [10]*100
                t = OutlierDetector(algo_name='modified_z_detector')
                t.fit(x)
                print t.predict(x)  # True means outlier
                print t.score(x)    # outlier scores


