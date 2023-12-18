import numpy as np
from ziptie.algo import Ziptie

n_inputs = 10
bundle_limit = 10
zt = Ziptie(n_inputs)

done = False
n_iter = 0
while not done:
    n_iter += 1
    print(f"{n_iter}", end="\r")

    inputs = np.random.sample(n_inputs)

    ##  step() is a convenience function that runs these three lines
    # zt.create_new_bundles()
    # zt.grow_bundles()
    # bundle_activities = zt.update_bundles(inputs)
    bundle_activities = zt.step(inputs)

    if zt.n_bundles >= bundle_limit:
        done = True

print(f"{zt.n_bundles} bundles in {n_iter} iterations.")
