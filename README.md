# Ziptie: An unsupervised feature learning algorithm

[Here's the full story](https://codeberg.org/brohrer/ziptie-paper/raw/branch/main/ziptie.pdf)
of how Ziptie works and why it was created.

## Installation

```bash
uv add ziptie
```

or

```bash
pip install ziptie
```

## Usage

### Import
```python
from ziptie.algo import Ziptie
```

### Initialize
```python
zt = Ziptie(n_inputs)
```

### Update agglomeration energies and create bundles on each iteration
```python
zt.create_new_bundles()
zt.grow_bundles()
```

### Calculate outputs on each iteration
```python
bundle_activities = zt.update_bundles(inputs)
```

### Take care of all of it at once
There is a convenience function that takes care of `create_new_bundles`,
`grow_bundles`, and `update_bundles`, if you prefer the shorthand.
```python
bundle_activities = zt.step(inputs)
```

### Impose a stopping condition
```python
zt.n_bundles >= bundle_limit
```

## Example

Putting it all together in a bare-bones example
(also in [example.py](ziptie/example.py)).

```python
import numpy as np
from ziptie.algo import Ziptie

n_inputs = 10
bundle_limit = 10
zt = Ziptie(n_inputs)

done = False
while not done:
    inputs = np.random.sample(n_inputs)
    bundle_activities = zt.step(inputs)

    if zt.n_bundles >= bundle_limit:
        done = True

print("Done!")
```

## Feature explanation

One trick Ziptie is good at is interpreting and explaining the features
it creates. Any collection of bundle activities can be projected back down
to the set of inputs that created it.

```python
inputs = zt.project_bundle_activities(bundle_activities)
```

To get a picture of a single feature, you can construct a
sparse `bundle_activities` array, with only a single non-zero element
for the feature you want to investigate.

## Benchmark

It's informative to run Ziptie on your own system with different numbers
of input cables to see how long it takes to run, and to see how those
per-iteration run times grow as the number of bundles increases.

```bash
uv run src/ziptie/benchmark.py
uv run src/ziptie/benchmark_plot.py
```

Don't be afraid to make changes to `benchmark.py` and see how it affects
the run times.

## Tweaking Ziptie's behavior through initialization arguments

There are a handful of constants and hyperparameters that allow
you to trade speed for accuracy and to adjust Ziptie's behaviors.

From the code:
```python
def __init__(
        self,
        n_cables=16,
        name='ziptie',
        activity_deadzone=.01,
        threshold=1e3,
        growth_threshold=None,
        growth_check_frequency=None,
        nucleation_check_frequency=None,
):
```

**`n_cables`** (int) has already been introduced. It is the number of cable
inputs the Ziptie expects.

**`activity_deadzone`** (float, default of .01) is the threshold
below which any cable or bundle activity will be snapped down to zero.
This helps maintain sparsity without otherwise changing the behavior much.

**`threshold`** (float, default of 1e3) is the agglomeration energy
threshold for creating a new bundle from two cables. Increasing this
means that bundles will form more slowly, but the are more likely to
capture the underlying relationships between cables.

**`growth_threshold`** (float, default of `None`) is the optional
argument for setting the cable-bundle agglomeration threshold separately.
If `None` it will use whatever value was supplied as the `threshold`.
Having different thresholds for cable-cable and cable-bundle agglomeration
can change whether the Ziptie tends to create more small bundles or fewer
larger ones.

**`nucleation_check_frequency`** (float, default of `None`) is roughly
how many time steps will pass between checking whether there is a pair
of cables that has accumulated enough agglomeration energy to become
a new bundle. This check is expensive so performing it less often is a
good way to speed up the Ziptie. If `None` is supplied, this defaults
to the agglomeration threshold / 10.

**`growth_check_frequency`** (float, default of `None`) is
similar to the `nucleation_check_frequency`, but for agglomeration of
cable-bundle bundles.
