"""
A handy script for test driving a Ziptie.
It's not a proper test integration test, but I use it as a half-hearted
smoke test to make sure the code will still turn over after I've
made some changes.
"""

import time
import numpy as np
from ziptie.algo import Ziptie

# How many times to repeat the overall benchmark loop.
# Run it multiple times if you'd like to build confidence in your times.
n_loops = 1

# How may time steps to walk through per loop.
max_iters = 1e6

# Total number of cables feeding into the Ziptie.
n_cables = 10000
# Number of those cables that will be non-zero on any given time step.
# This is related to the l0 sparsity of the inputs.
# Fully dense would be n_active_cables = n_cables
n_active_cables = int(n_cables / 20)

iter_times = []
iter_data = []

for i_pass in range(n_loops):
    zt = Ziptie(n_cables)

    # "Warm up" the Ziptie.
    # Run through it once to trigger all the just-in-time compiling.
    inputs = np.random.sample(n_cables)
    bundle_activities = zt.step(inputs)

    total_time = 0
    i_iter = 0
    while i_iter < max_iters:
        i_iter += 1
        print(f"{i_iter}", end="\r")

        i_active = np.random.choice(n_cables, size=n_active_cables, replace=False)
        inputs[i_active] = np.random.sample(n_active_cables)

        # Isolate the Ziptie's operation time each step and
        # measure it.
        start = time.time()
        bundle_activities = zt.step(inputs)
        elapsed = time.time() - start

        iter_data.append([elapsed, n_cables, zt.n_bundles])
        total_time += elapsed

    us_per_iter = int(1e6 * total_time / max_iters)
    iter_times.append(us_per_iter)

    print(f"Pass {i_pass}, {zt.n_bundles} bundles, {us_per_iter}us per iter")

print(f"{int(np.mean(np.array(iter_times)))} usec per iteration")
print(f"{int(np.std(np.array(iter_times)))} usec standard deviation")
print(f"{int(np.std(np.array(iter_times)) / n_loops)} usec standard error")

# Save the observed iteration times out for visualization and analysis.
np.save("bench_data.npy", np.array(iter_data))
