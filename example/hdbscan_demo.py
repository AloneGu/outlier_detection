# generate sample data
from sklearn.datasets.samples_generator import make_blobs
import numpy as np

blobs, labels = make_blobs(n_samples=200, n_features=2)
random_idx = np.random.randint(0, 200, size=10, dtype=int)
blobs[random_idx] = blobs[random_idx] * 10 + 5

test_score = [[18, 18], [19, 19]]

# create model
from outlier_detector.base_detector import OutlierDetector

cluster = OutlierDetector(algo_name='hdbscan')

# fit
cluster.fit(blobs)
res = cluster.predict(blobs)
print cluster.score(test_score)
print cluster.score(blobs)

# plot
import matplotlib.pyplot as plt

plt.figure()
x = [r[0] for r in blobs]
y = [r[1] for r in blobs]
c = ['r' if r == True else 'g' for r in res]
plt.scatter(x, y, color=c)
plt.show()
