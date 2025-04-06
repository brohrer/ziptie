"""
Plot the iteration times collected by benchmark.py in a scatter plot.
"""

import numpy as np
import matplotlib.pyplot as plt

bench_data = np.load("bench_data.npy")
times = bench_data[:, 0]
cables = bench_data[:, 1]
bundles = bench_data[:, 2]
n_pts = bench_data.shape[0]
i_pts = np.arange(0, n_pts, 30)
plt.scatter(bundles[i_pts], 1e6 * times[i_pts], s=2, alpha=0.3)
plt.xlabel("Number of bundles")
plt.ylabel("$\mu$ sec per iteration")
plt.show()
