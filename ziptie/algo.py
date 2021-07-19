import os
from numba import njit
import numpy as np


class Ziptie(object):
    """
    An incremental unsupervised clustering algorithm.

    Input channels are clustered together into mutually co-active sets.
    A helpful metaphor is bundling cables together with zip ties.
    Cables that carry related signals are commonly co-active,
    and are grouped together. Cables that are co-active with existing
    bundles can be added to those bundles. A single cable may be ziptied
    into several different bundles. Co-activity is estimated
    incrementally, that is, the algorithm updates the estimate after
    each new set of signals is received.

    When stacked with other levels,
    zipties form a deep sparse network.
    This network has the extremely desirable characteristic of
    l-0 sparsity--the number of non-zero weights is minimized.
    The vast majority of weights in this network are zero,
    and the rest are one.
    This makes sparse computation feasible and allows for
    straightforward interpretation and visualization of the
    features.
    """
    def __init__(
            self,
            n_cables=16,
            name='ziptie',
            threshold=1e3,
    ):
        """
        Initialize the ziptie, pre-allocating data structures.

        Parameters
        ----------
        n_cables : int
            The number of inputs to the Ziptie.
        name : str, optional
            The name assigned to this instance of the Ziptie algorithm.
        threshold : float
            The point at which to nucleate a new bundle or
            agglomerate to an existing one.
        """
        self.name = name
        self.n_cables = n_cables
        # n_bundles : int, optional
        #     The number of bundle outputs from the Ziptie.
        self.n_bundles = 0

        # nucleation_threshold : float
        #     Threshold above which nucleation energy results in nucleation.
        self.nucleation_threshold = threshold
        # agglomeration_threshold
        #     Threshold above which agglomeration energy results
        #     in agglomeration.
        self.agglomeration_threshold = self.nucleation_threshold
        # activity_threshold : float
        #     Threshold below which input activity is teated as zero.
        #     By ignoring the small activity values,
        #     computation gets much faster.
        self.activity_threshold = .01
        # cable_activities : array of floats
        #     The current set of input actvities.
        self.cable_activities = np.zeros(self.n_cables)
        # bundle_activities : array of floats
        #     The current set of bundle activities.
        self.bundle_activities = np.zeros(self.n_bundles)

        # mapping: 2D array of ints
        #     The mapping between cables and bundles.
        #     If element i,j is 1, then cable i is a member of bundle j
        self.mapping = np.zeros(
            (self.n_cables, self.n_cables), dtype=np.int)

        # agglomeration_energy: 2D array of floats
        #     The accumulated agglomeration energy for each
        #     bundle-cable pair. Bundles are represented in rows,
        #     cables are in columns.
        self.agglomeration_energy = np.zeros(
            (self.n_cables, self.n_cables))
        # agglomeration_mask: 2D array of floats
        #     A binary array indicating which cable-bundle
        #     pairs are allowed to accumulate
        #     energy and which are not. Some combinations are
        #     disallowed because they result in redundant bundles.
        self.agglomeration_mask = np.ones(
            (self.n_cables, self.n_cables))
        # nucleation_energy: 2D array of floats
        #     The accumualted nucleation energy associated
        #     with each cable-cable pair.
        self.nucleation_energy = np.zeros(
            (self.n_cables, self.n_cables))
        # nucleation_mask: 2D array of floats
        #     A binary array indicating which cable-cable
        #     pairs are allowed to accumulate
        #     energy and which are not. Some combinations are
        #     disallowed because they result in redundant bundles.
        #     Make the diagonal zero to disallow cables to pair with
        #     themselves.
        self.nucleation_mask = (
            np.ones((self.n_cables, self.n_cables))
            - np.eye(self.n_cables))

    def update_bundles_fast(self, new_cable_activities):
        """
        Calculate how much the cables' activities contribute to each bundle.
        Find bundle activities by taking the minimum input value
        in the set of cables in the bundle.

        Parameters
        ----------
        new_cable_activities: array of floats

        Returns
        -------
        bundle_activities: array of floats
        """
        self.cable_activities = new_cable_activities
        self.bundle_activities = np.zeros(self.n_bundles)
        for i_bundle in np.unique(np.where(self.mapping)[1]):
            i_cables = np.where(self.mapping[:, i_bundle])[0]
            self.bundle_activities[i_bundle] = np.min(
                self.cable_activities[i_cables])
        return self.bundle_activities

    def update_bundles(self, new_cable_activities):
        """
        Calculate how much the cables' activities contribute to each bundle.
        Find bundle activities by taking the minimum input value
        in the set of cables in the bundle.

        Parameters
        ----------
        new_cable_activities: array of floats

        Returns
        -------
        bundle_activities: array of floats
        """
        self.cable_activities = new_cable_activities
        self.bundle_activities = np.zeros(self.n_bundles)
        update_bundles_numba(
            self.cable_activities,
            self.bundle_activities,
            self.activity_threshold,
            self.n_bundles,
            self.mapping)
        return self.bundle_activities

    def create_new_bundles(self):
        """
        If the right conditions have been reached, create a new bundle.
        """
        # Incrementally accumulate nucleation energy.
        nucleation_energy_gather(
            self.cable_activities,
            self.nucleation_energy,
            self.nucleation_mask,
        )
        max_energy, i_cable_a, i_cable_b = max_2d(
            self.nucleation_energy)

        # Add a new bundle if appropriate
        if max_energy > self.nucleation_threshold:
            i_bundle = self.n_bundles
            self.increment_n_bundles()
            self.mapping[i_cable_a, i_bundle] = 1
            self.mapping[i_cable_b, i_bundle] = 1

            # Reset the accumulated nucleation and agglomeration energy
            # for the two cables involved.
            self.nucleation_energy[i_cable_a, :] = 0
            self.nucleation_energy[i_cable_b, :] = 0
            self.nucleation_energy[:, i_cable_a] = 0
            self.nucleation_energy[:, i_cable_b] = 0
            self.agglomeration_energy[:, i_cable_a] = 0
            self.agglomeration_energy[:, i_cable_b] = 0

            # Update nucleation_mask to prevent the two cables from
            # accumulating nucleation energy in the future.
            self.nucleation_mask[i_cable_a, i_cable_b] = 0
            self.nucleation_mask[i_cable_b, i_cable_a] = 0

            # Update agglomeration_mask to account for the new bundle.
            # The new bundle should not accumulate agglomeration energy
            # with any of the cables that any of its constituent cables
            # are blocked from nucleating with.
            blocked_a = np.where(self.nucleation_mask[i_cable_a, :] == 0)[0]
            blocked_b = np.where(self.nucleation_mask[i_cable_b, :] == 0)[0]
            blocked = np.union1d(blocked_a, blocked_b)
            self.agglomeration_mask[i_bundle, blocked] = 0

    def grow_bundles(self):
        """
        Update an estimate of co-activity between all cables.
        """
        # Incrementally accumulate agglomeration energy.
        agglomeration_energy_gather(
            self.bundle_activities,
            self.cable_activities,
            self.n_bundles,
            self.agglomeration_energy,
            self.agglomeration_mask,
        )
        max_energy, i_bundle, i_cable = max_2d(
            self.agglomeration_energy)

        # Add a new bundle if appropriate
        if max_energy > self.agglomeration_threshold:
            # Add the new bundle to the end of the list.
            i_new_bundle = self.n_bundles
            self.increment_n_bundles()

            # Make a copy of the growing bundle.
            self.mapping[:, i_new_bundle] = self.mapping[:, i_bundle]
            # Add in the new cable.
            self.mapping[i_cable, i_bundle] = 1

            # Reset the accumulated nucleation and agglomeration energy
            # for the two cables involved.
            self.nucleation_energy[i_cable, :] = 0
            self.nucleation_energy[i_cable, :] = 0
            self.nucleation_energy[:, i_cable] = 0
            self.nucleation_energy[:, i_cable] = 0
            self.agglomeration_energy[:, i_cable] = 0
            self.agglomeration_energy[i_bundle, :] = 0

            # Update agglomeration_mask to account for the new bundle.
            # The new bundle should not accumulate agglomeration energy with
            # 1) the cables that its constituent cable
            #    are blocked from nucleating with or
            # 2) the cables that its constituent bundle
            #    are blocked from agglomerating with.
            blocked_cable = np.where(
                self.nucleation_mask[i_cable, :] == 0)
            blocked_bundle = np.where(
                self.agglomeration_mask[i_bundle, :] == 0)
            blocked = np.union1d(blocked_cable[0], blocked_bundle[0])
            self.agglomeration_mask[i_new_bundle, blocked] = 0

    def update_inputs(self, resets):
        """
        Reset indicated cables and all the bundles associated with them.

        Parameters
        ----------
        resets: array of ints
            The indices of the cables that are being reset

        Returns
        -------
        upstream_resets: array of ints
            The indices of the bundles to be reset.
        """

        upstream_resets = []
        for i_cable in resets:
            for i_bundle in np.where(self.mapping[i_cable, :])[0]:
                upstream_resets.append(i_bundle)
                # Remove the bundle from the mappings in both directions.
                self.mapping[:, i_bundle] = 0

                self.agglomeration_mask[i_bundle, :] = 1
                self.agglomeration_energy[i_bundle, :] = 0

            self.mapping[i_cable, :] = 0

            self.agglomeration_mask[:, i_cable] = 1
            self.agglomeration_energy[:, i_cable] = 0

            self.nucleation_mask[i_cable, :] = 1
            self.nucleation_mask[:, i_cable] = 1
            self.nucleation_mask[i_cable, i_cable] = 0
            self.nucleation_energy[i_cable, :] = 0
            self.nucleation_energy[:, i_cable] = 0

        return upstream_resets

    def increment_n_bundles(self):
        """
        Add one to n_map entries and grow the bundle map as needed.
        """
        self.n_bundles += 1
        n_max_bundles = self.agglomeration_energy.shape[0]
        if self.n_bundles >= n_max_bundles:
            new_max_bundles = n_max_bundles * 2
            new_agglomeration_energy = np.zeros(
                (new_max_bundles, self.n_cables))
            new_agglomeration_energy[:n_max_bundles, :] = (
                    self.agglomeration_energy)
            self.agglomeration_energy = new_agglomeration_energy

            new_agglomeration_mask = np.zeros(
                (new_max_bundles, self.n_cables))
            new_agglomeration_mask[:n_max_bundles, :] = (
                    self.agglomeration_mask)
            self.agglomeration_mask = new_agglomeration_mask

            new_mapping = np.zeros((self.n_cables, new_max_bundles))
            new_mapping[:, :n_max_bundles] = self.mapping
            self.mapping = new_mapping

    def get_index_projection(self, i_bundle):
        """
        Project i_bundle down to its cable indices.

        Parameters
        ----------
        i_bundle : int
            The index of the bundle to project onto its constituent cables.

        Returns
        -------
        projection : array of floats
            An array of zeros and ones, representing all the cables that
            contribute to the bundle. The values projection
            corresponding to all the cables that contribute are 1.
        """
        projection = np.zeros(self.n_cables)
        projection[self.get_index_projection_cables()] = 1
        return projection

    def get_index_projection_cables(self, i_bundle):
        """
        Project i_bundle down to its cable indices.

        Parameters
        ----------
        i_bundle : int
            The index of the bundle to project onto its constituent cables.

        Returns
        -------
        projection_indices : array of ints
            An array of cable indices, representing all the cables that
            contribute to the bundle.
        """
        return np.where(self.mapping[:, i_bundle])[0]

    def project_bundle_activities(self, bundle_activities):
        """
        Take a set of bundle activities and project them to cable activities.

        Parameters
        ----------
        bundle_activities: array of floats

        Results
        -------
        cable_activities: array of floats
        """
        cable_activities = np.zeros(self.n_cables)
        for i_cable in np.unique(np.where(self.mapping)[0]):
            i_bundles = np.where(self.mapping[i_cable, :])[0]
            cable_activities[i_cable] = np.max(bundle_activities[i_bundles])
        return cable_activities


@njit
def max_2d(array2d):
    """
    Find the maximum value of a dense 2D array, with its row and column

    Parameters
    ----------
    array2d : 2D array of floats
        The array to find the maximum value of.

    Returns
    -------
    Results are returned indirectly by modifying results.
    The results array has three elements and holds
    max_val: float
        The maximum value found
    i_row, i_col: ints
        The row and column in which it was found
    """
    max_val = 0
    i_row_max = -1
    i_col_max = -1
    n_rows, n_cols = array2d.shape
    for i_row in range(n_rows):
        for i_col in range(n_cols):
            if array2d[i_row, i_col] > max_val:
                max_val = array2d[i_row, i_col]
                i_row_max = i_row
                i_col_max = i_col

    return (max_val, i_row_max, i_col_max)


@njit
def update_bundles_numba(
    cable_activities,
    bundle_activities,
    activity_threshold,
    n_bundles,
    mapping,
):
    # Get a list of all the bundles, from most recently created
    # to the oldest.
    i_bundles = np.sort(np.unique(np.where(mapping)[1]))[::-1]

    remaining_cable_activities = cable_activities.copy()
    for i_bundle in i_bundles:
        i_cables = np.where(mapping[:, i_bundle])[0]
        bundle_activity = np.min(remaining_cable_activities[i_cables])
        if bundle_activity > activity_threshold:
            bundle_activities[i_bundle] = bundle_activity
            remaining_cable_activities[i_cables] -= bundle_activity


@njit
def nucleation_energy_gather(
    cable_activities,
    nucleation_energy,
    nucleation_mask,
):
    """
    Gather nucleation energy.

    This formulation takes advantage of loops and the sparsity of the data.
    The original arithmetic looks like
        nucleation_energy += (cable_activities *
                              cable_activities.T *
                              nucleation_energy_rate)
    Parameters
    ----------
    cable_activities : array of floats
        The current activity of each input feature.
    nucleation_energy : 2D array of floats
        The amount of nucleation energy accumulated between each pair of
        input features.
    nucleation_mask: 2D array of floats
        A mask showing which input-input pairs are allowed
        to accumulate energy.

    Results
    -------
    Returned indirectly by modifying nucleation_energy.
    """
    for i_cable1, activity1 in enumerate(cable_activities):
        if activity1 > 0:
            for i_cable2, _ in enumerate(cable_activities):
                activity2 = cable_activities[i_cable2]
                if activity2 > 0:
                    if nucleation_mask[i_cable1, i_cable2]:
                        nucleation_energy[i_cable1, i_cable2] += (
                            activity1 * activity2)


@njit
def agglomeration_energy_gather(
    bundle_activities,
    cable_activities,
    n_bundles,
    agglomeration_energy,
    agglomeration_mask,
):
    """
    Accumulate the energy binding a new feature to an existing bundle.

    This formulation takes advantage of loops and the sparsity of the data.
    The original arithmetic looks like
        coactivities = bundle_activities * cable_activities.T
        agglomeration_energy += coactivities

    Parameters
    ----------
    bundle_activities : array of floats
        The activity level of each bundle.
    cable_activities : array of floats
        The current activity of each input feature that is not explained by
        or captured in a bundle.
    n_bundles : int
        The number of bundles that have been created so far.
    agglomeration_energy : 2D array of floats
        The total energy that has been accumulated between each input feature
        and each bundle.
    agglomeration_mask: 2D array of floats
        A mask showing which input-input pairs are allowed
        to accumulate energy.

    Results
    -------
    Returned indirectly by modifying agglomeration_energy.
    """
    for i_cable, activity in enumerate(cable_activities):
        if activity > 0:
            # Only decay bundles that have been created
            for i_bundle in range(n_bundles):
                if bundle_activities[i_bundle] > 0:
                    if agglomeration_mask[i_bundle, i_cable]:
                        coactivity = activity * bundle_activities[i_bundle]
                        agglomeration_energy[i_bundle, i_cable] += coactivity
