from numba import njit
import numpy as np


class Ziptie:
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
        name="ziptie",
        activity_deadzone=0.01,
        threshold=1e3,
        growth_threshold=None,
        growth_check_frequency=None,
        nucleation_check_frequency=None,
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
            The agglomeration energy threshold, above which
            to nucleate a new bundle.
        growth_threshold : float
            The agglomeration energy threshold, above which
            to agglomerate a cable to an existing one.
        """
        self.name = name
        self.n_cables = n_cables
        # n_bundles : int, optional
        #     The number of bundle outputs from the Ziptie.
        self.n_bundles = 0

        # nucleation_threshold : float
        #     Threshold above which nucleation energy results in
        #     nucleating a bundle--combining two cables
        #     to create a new bundle.
        self.nucleation_threshold = threshold
        # agglomeration_threshold
        #     Threshold above which agglomeration energy results
        #     growing a bundle--combinig a cable and a bundle
        #     to create a new bundle.
        #     The nucleation_threshold and agglomeration_threshold
        #     don't have to be the same. They can be varied independently.
        #     to generate different agglomeration behaviors that favor
        #     either more small bundles, or fewer larger ones.
        if growth_threshold is None:
            self.agglomeration_threshold = self.nucleation_threshold
        else:
            self.agglomeration_threshold = growth_threshold

        # activity_deadzone : float
        #     Threshold below which input activity is teated as zero.
        #     By ignoring the small activity values,
        #     computation gets much faster.
        self.activity_deadzone = activity_deadzone
        # cable_activities : array of floats
        #     The current set of input actvities.
        self.cable_activities = np.zeros(self.n_cables)
        # remaining_cable_activities : array of floats
        #     The set of input actvities not associated with any bundle.
        self.cable_activities = np.zeros(self.n_cables)
        # bundle_activities : array of floats
        #     The current set of bundle activities.
        self.bundle_activities = np.zeros(self.n_bundles)

        # mapping: 2D array of ints
        #     The mapping between cables and bundles.
        #     Each row represents a single bundle.
        #     It contains the indices of all the cables in that bundle.
        #     Unneeded elements are padded out with -1.
        self.mapping = -np.ones((self.n_cables, self.n_cables), dtype=int)
        # n_cables_by_bundle: 1D array of ints
        #     The number of cables in each bundle.
        #     Matches the corresponding bundle indices in self.mapping
        #     and the agglomeration energy arrays.
        self.n_cables_by_bundle = np.zeros(self.n_cables, dtype=int)

        # agglomeration_energy: 2D array of floats
        #     The accumulated agglomeration energy for each
        #     bundle-cable pair. Bundles are represented in rows,
        #     cables are in columns.
        self.agglomeration_energy = np.zeros((self.n_cables, self.n_cables))
        # agglomeration_mask: 2D array of floats
        #     A binary array indicating which cable-bundle
        #     pairs are allowed to accumulate
        #     energy and which are not. Some combinations are
        #     disallowed because they result in redundant bundles.
        self.agglomeration_mask = np.ones((self.n_cables, self.n_cables), dtype=int)
        # nucleation_energy: 2D array of floats
        #     The accumualted nucleation energy associated
        #     with each cable-cable pair.
        self.nucleation_energy = np.zeros((self.n_cables, self.n_cables))
        # nucleation_mask: 2D array of floats
        #     A binary array indicating which cable-cable
        #     pairs are allowed to accumulate
        #     energy and which are not. Some combinations are
        #     disallowed because they result in redundant bundles.
        #     Make the diagonal zero to disallow cables to pair with
        #     themselves.
        self.nucleation_mask = np.ones(
            (self.n_cables, self.n_cables), dtype=int
        ) - np.eye(self.n_cables, dtype=int)

        # nucleation_check_frequency: floats
        if nucleation_check_frequency is None:
            self.nucleation_check_fraction = 10.0 / self.nucleation_threshold
        else:
            self.nucleation_check_fraction = 1 / nucleation_check_frequency

        if growth_check_frequency is None:
            self.agglomeration_check_fraction = 10.0 / self.agglomeration_threshold
        else:
            self.agglomeration_check_fraction = 1 / growth_check_frequency

    def step(self, inputs):
        """
        A convenience function to run the three methods that are typically
        run at every step.
        """
        self.create_new_bundles()
        self.grow_bundles()
        return self.update_bundles(inputs)

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
        self.remaining_cable_activities = self.cable_activities.copy()
        update_bundles_numba(
            self.remaining_cable_activities,
            self.bundle_activities,
            self.activity_deadzone,
            self.n_bundles,
            self.mapping,
            self.n_cables_by_bundle,
        )
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

        if np.random.sample() > self.nucleation_check_fraction:
            return

        i_cable_a, i_cable_b = threshold_check(
            self.nucleation_energy, self.nucleation_threshold
        )
        # max_energy, i_cable_a, i_cable_b = max_2d(
        #     self.nucleation_energy)

        # Add a new bundle if appropriate
        # if max_energy > self.nucleation_threshold:
        if i_cable_a > -1:
            i_bundle = self.n_bundles
            self.increment_n_bundles()
            self.n_cables_by_bundle[i_bundle] = 2
            self.mapping[i_bundle, :2] = np.array([i_cable_a, i_cable_b], dtype=int)

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
        Update an estimate of co-activity between all cables and bundles.
        """
        # Incrementally accumulate agglomeration growth energy.
        # import time
        # start_agg = time.time()
        agglomeration_energy_gather(
            self.bundle_activities,
            self.cable_activities,
            self.n_bundles,
            self.agglomeration_energy,
            self.agglomeration_mask,
        )
        # elapsed_agg = time.time() - start_agg

        if np.random.sample() > self.agglomeration_check_fraction:
            return

        if self.n_bundles == 0:
            return

        # start_max = time.time()
        # TODO time this
        i_bundle, i_cable = threshold_check(
            self.agglomeration_energy[: self.n_bundles, :], self.agglomeration_threshold
        )

        # max_energy, i_bundle, i_cable = max_2d(
        #     self.agglomeration_energy)

        # i_max = np.argmax(self.agglomeration_energy[:self.n_bundles, :])
        # n_bund, n_cab = self.agglomeration_energy.shape
        # i_bundle, i_cable = np.unravel_index(i_max, self.agglomeration_energy.shape)
        # i_bundle = int(i_max / n_cab)
        # i_cable = i_max % n_cab
        # max_energy = self.agglomeration_energy[i_bundle, i_cable]

        # elapsed_max = time.time() - start_agg
        # print(f"{elapsed_max / elapsed_agg}")

        # Add a new bundle if appropriate
        # if max_energy > self.agglomeration_threshold:
        if i_bundle > -1:
            # Add the new bundle to the end of the list.
            i_new_bundle = self.n_bundles
            self.increment_n_bundles()

            # Get the number of cables in the constituent bundle.
            n_cables_old = self.n_cables_by_bundle[i_bundle]
            self.n_cables_by_bundle[i_new_bundle] = n_cables_old + 1
            # Make a copy of the growing bundle.
            self.mapping[i_new_bundle, :n_cables_old] = self.mapping[
                i_bundle, :n_cables_old
            ]
            # Add in the new cable.
            self.mapping[i_new_bundle, n_cables_old] = i_cable

            # Reset the accumulated nucleation and agglomeration energy
            # for the cable and bundle involved.
            self.nucleation_energy[i_cable, :] = 0
            self.nucleation_energy[:, i_cable] = 0
            self.agglomeration_energy[:, i_cable] = 0
            self.agglomeration_energy[i_bundle, :] = 0

            # Update agglomeration_mask to prevent the cable and bundle from
            # accumulating agglomeration energy in the future.
            self.agglomeration_mask[i_bundle, i_cable] = 0

            # Update agglomeration_mask to account for the new bundle.
            # The new bundle should not accumulate agglomeration energy with
            # 1) the cables that its constituent cable
            #    are blocked from nucleating with or
            # 2) the cables that its constituent bundle
            #    are blocked from agglomerating with.
            blocked_cable = np.where(self.nucleation_mask[i_cable, :] == 0)
            blocked_bundle = np.where(self.agglomeration_mask[i_bundle, :] == 0)
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
            for i_bundle in range(self.n_bundles):
                n_cables = self.n_cables_by_bundle[i_bundle]
                if i_cable in self.mapping[i_bundle, :n_cables]:
                    upstream_resets.append(i_bundle)
                    # Remove the bundle from the mappings in both directions.
                    self.mapping[i_bundle, :n_cables] = 0

                    self.agglomeration_mask[i_bundle, :] = 1
                    self.agglomeration_energy[i_bundle, :] = 0

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
        new_bundle_activities = np.zeros(self.n_bundles)
        new_bundle_activities[:-1] = self.bundle_activities
        self.bundle_activities = new_bundle_activities
        bundle_capacity = self.agglomeration_energy.shape[0]
        if self.n_bundles >= bundle_capacity:
            new_bundle_capacity = bundle_capacity * 2
            new_agglomeration_energy = np.zeros((new_bundle_capacity, self.n_cables))
            new_agglomeration_energy[:bundle_capacity, :] = self.agglomeration_energy
            self.agglomeration_energy = new_agglomeration_energy

            new_agglomeration_mask = np.zeros(
                (new_bundle_capacity, self.n_cables), dtype=int
            )
            new_agglomeration_mask[:bundle_capacity, :] = self.agglomeration_mask
            self.agglomeration_mask = new_agglomeration_mask

            new_mapping = -np.ones((new_bundle_capacity, self.n_cables), dtype=int)
            new_mapping[:bundle_capacity, :] = self.mapping
            self.mapping = new_mapping

            new_n_cables_by_bundle = np.zeros(new_bundle_capacity, dtype=int)
            new_n_cables_by_bundle[:bundle_capacity] = self.n_cables_by_bundle
            self.n_cables_by_bundle = new_n_cables_by_bundle

        # Check whether more cable capacity is needed on the bundles.
        bundle_capacity, cable_capacity = self.mapping.shape
        n_max_cables_by_bundle = np.max(self.n_cables_by_bundle)

        if n_max_cables_by_bundle >= cable_capacity:
            new_cable_capacity = cable_capacity * 2

            new_mapping = -np.ones((bundle_capacity, new_cable_capacity), dtype=int)
            new_mapping[:, :cable_capacity] = self.mapping
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
        n_cables = self.n_cables_by_bundle[i_bundle]
        projection_indices = self.mapping[i_bundle, :n_cables]
        return projection_indices

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
        for i_bundle in range(self.n_bundles):
            # for i_cable in np.unique(np.where(self.mapping)[0]):
            n_cables = self.n_cables_by_bundle[i_bundle]
            for i_cable in self.mapping[i_bundle, :n_cables]:
                cable_activities[i_cable] = np.maximum(
                    cable_activities[i_cable], bundle_activities[i_bundle]
                )
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
def threshold_check(array2d, threshold):
    """
    TODO: fix
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
    n_rows, n_cols = array2d.shape
    for i_row in range(n_rows):
        for i_col in range(n_cols):
            if array2d[i_row, i_col] > threshold:
                return (i_row, i_col)

    return (-1, -1)


@njit
def update_bundles_numba(
    cable_activities,
    bundle_activities,
    activity_deadzone,
    n_bundles,
    mapping,
    n_cables_by_bundle,
):
    # Get a list of all the bundles, from most recently created
    # to the oldest.
    for i_bundle in range(n_bundles - 1, -1, -1):
        # Find the set of cable indices in this bundle.
        n_cables = n_cables_by_bundle[i_bundle]
        i_cables = mapping[i_bundle, :n_cables]

        # Find the bundle activity.
        # It is the minimum of all of its constituent cables' activities.
        bundle_activity = 1
        for i_cable in i_cables:
            cable_activity = cable_activities[i_cable]
            if cable_activity < bundle_activity:
                bundle_activity = cable_activity

        # Only if the bundle activity is significant, keep it.
        if bundle_activity > activity_deadzone:
            bundle_activities[i_bundle] = bundle_activity

            # Reduce all the constituent cables' activities by the
            # amount of the bundle's activity.
            for i_cable in i_cables:
                cable_activities[i_cable] -= bundle_activity

    # As a final step, if any individual cables' activities are
    # not large enough to matter, set them to zero so that downstream
    # calculations can be streamlined.
    for i_cable, activity in enumerate(cable_activities):
        if activity < activity_deadzone:
            cable_activities[i_cable] = 0


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
        nucleation_energy += (
            (cable_activities @ cable_activities.T) *
            nucleation_energy_mask
        )
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
    # Use enumerate instead of range here for a small performance boost.
    # I'm not sure why, but I think it's related to the default type
    # assigned to the counter.
    # Also, believe it or not, there's about a ~5% speedup
    # From * not * pulling the cable_activity value from the enumerate,
    # but instead pulling it from the array using the index on the next line.
    # My guess is that it avoids some automatic type checking or
    # conversion or something like that.
    for i_cable_1, _ in enumerate(cable_activities):
        activity_1 = cable_activities[i_cable_1]
        # Only populate the lower half of the array.
        # It is symmetric, and doing this removes the redundancy.
        # It doesn't change the overall behavior of the Ziptie,
        # but it does buy a ~20% speed boost for this function.
        for i_cable_2, _ in enumerate(cable_activities[:i_cable_1]):
            activity_2 = cable_activities[i_cable_2]
            coactivity = activity_1 * activity_2

            mask = nucleation_mask[i_cable_1, i_cable_2]
            nucleation_energy[i_cable_1, i_cable_2] += coactivity * mask


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
    for i_bundle, _ in enumerate(bundle_activities[:n_bundles]):
        bundle_activity = bundle_activities[i_bundle]
        if bundle_activity > 0:
            for i_cable, _ in enumerate(cable_activities):
                cable_activity = cable_activities[i_cable]
                coactivity = cable_activity * bundle_activity
                mask = agglomeration_mask[i_bundle, i_cable]
                agglomeration_energy[i_bundle, i_cable] += coactivity * mask
