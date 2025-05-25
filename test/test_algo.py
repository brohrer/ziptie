import numpy as np
from ziptie.algo import Ziptie


def test_test():
    assert True


def test_defaults():
    zt = Ziptie()
    assert zt.n_cables == 16
    assert zt.n_bundles_max == 64
    assert zt.nucleation_threshold == 1e3


def test_sizes():
    n_cables = 77
    n_bundles_max = 173
    zt = Ziptie(
        n_cables=n_cables,
        n_bundles_max=n_bundles_max,
    )
    assert zt.mapping.shape[0] == n_bundles_max
    assert zt.mapping.shape[1] == n_cables
    assert zt.nucleation_energy.shape[0] == n_cables
    assert zt.nucleation_energy.shape[1] == n_cables
    assert zt.agglomeration_energy.shape[0] == n_bundles_max
    assert zt.agglomeration_energy.shape[1] == n_cables


def test_nucleation_energy_gathering():
    n_cables = 63
    n_bundles_max = 133
    zt = Ziptie(
        n_cables=n_cables,
        n_bundles_max=n_bundles_max,
    )
    inputs = np.ones(n_cables)
    for _ in range(2):
        zt.step(inputs)

    assert np.sum(zt.nucleation_energy) > 0
    assert np.sum(zt.agglomeration_energy) == 0


def test_agglomeration_energy_gathering():
    n_cables = 38
    n_bundles_max = 83
    zt = Ziptie(
        n_cables=n_cables,
        n_bundles_max=n_bundles_max,
    )
    inputs = np.ones(n_cables)
    while True:
        zt.step(inputs)
        if zt.n_bundles > 2:
            break

    assert np.sum(zt.agglomeration_energy) > 0


def test_bundle_creation():
    n_cables = 32
    n_bundles_max = 53
    zt = Ziptie(
        n_cables=n_cables,
        n_bundles_max=n_bundles_max,
    )
    inputs = np.ones(n_cables)
    while True:
        zt.step(inputs)
        if zt.n_bundles > 0:
            break

    assert zt.mapping[0][0] >= 0
    assert zt.mapping[0][1] >= 0
    assert zt.n_cables_by_bundle[0] == 2
    assert zt.nucleation_mask[zt.mapping[0][0], zt.mapping[0][1]] == 0
    assert zt.agglomeration_mask[0, zt.mapping[0][1]] == 0
