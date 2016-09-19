# create test data
from scipy import stats

x = stats.poisson.rvs(1, 0, 100)
noise = [16, 17, 18, 19, 10]

# fit and test
from outlier_detector.base_detector import OutlierDetector

detector = OutlierDetector(algo_name='poisson', param_dict={'cdf_thres': 0.9})
detector.fit(x)
print detector.predict(x[-5:])
print detector.score(x[-5:])
print detector.predict(noise)
print detector.score(noise)
