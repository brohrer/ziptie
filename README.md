# Ziptie: An unsupervised feature learning algorithm

[Here's the full story](https://codeberg.org/brohrer/ziptie-paper/raw/branch/main/ziptie.pdf)
of how Ziptie works and why it was created.

## Installation

First download the code.
In a terminal with a bash-like scripting language this can be done with

```
git clone https://codeberg.org/brohrer/ziptie.git
```

Then Python's package manager pip is a good way to get it into your
environment.

```python
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

### Update clusters on each iteration
```python
zt.create_new_bundles()
zt.grow_bundles()
```

### Calculate outputs on each iteration
```python
bundle_activities = zt.update_bundles(inputs)
```

### Impose a stopping condition
```python
zt.n_bundles >= bundle_limit
```
