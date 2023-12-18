# Ziptie: An unsupervised feature learning algorithm

[Here's the full story](https://codeberg.org/brohrer/ziptie-paper/raw/branch/main/ziptie.pdf)
of how Ziptie works and why it was created.

## Installation

First download the code.
In a terminal with a bash-like scripting language this can be done with

```bash
git clone https://codeberg.org/brohrer/ziptie.git
```

Then Python's package manager pip is a good way to get it into your
environment.

```bash
python3 -m pip install -e ziptie
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
python3 benchmark.py
python3 benchmark_plot.py
```

Don't be afraid to make changes to `benchmark.py` and see how it affects
the run times.

## Tweaking Ziptie's behavior through initialization arguments

There are a handful of constants and hyperparameters that allow
you to trade speed for accuracy and to adjust Ziptie's behaviors.


activity_deadzone=.01,
threshold=1e3,
growth_threshold=None,
growth_check_frequency=None,
nucleation_check_frequency=None,

