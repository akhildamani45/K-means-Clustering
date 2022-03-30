"""
Primary algorithm for k-Means clustering

This file contains the Algorithm class for performing k-means clustering.  While it is
the last part of the assignment, it is the heart of the clustering algorithm.  You
need this class to view the complete visualizer.

Akhil Damani ad674
11/17/21
"""
import math
import random
import numpy


# For accessing the previous parts of the assignment
import a6dataset
import a6cluster

# Part A
def valid_seeds(value, size):
    """
    Returns True if value is a valid list of seeds for clustering.

    A list of seeds is a k-element list OR tuple of integers between 0 and size-1.
    In addition, no seed element can appear twice.

    Parameter value: a value to check
    Precondition: value can be anything

    Paramater size: The database size
    Precondition: size is an int > 0
    """
    assert type(size) == int
    assert size > 0

    if type(value) == tuple or type(value) == list:
        if len(value) == 0:
            return True
        for num in range(len(value)):
            if (type(value[num]) == int and value.count(value[num]) == 1
            and 0 <= value[num] <= size - 1):
                result = True
            else:
                return False
    else:
        return False

    return result


class Algorithm(object):
    """
    A class to manage and run the k-means algorithm.

    The method step() performs one step of the calculation.  The method run() will
    continue the calculation until it converges (or reaches a maximum number of steps).
    """
    # IMMUTABLE ATTRIBUTES (Fixed after initialization with no DIRECT access)
    # Attribute _dataset: The Dataset for this algorithm
    # Invariant: _dataset is an instance of Dataset
    #
    # Attribute _cluster: The clusters to use at each step
    # Invariant: _cluster is a non-empty list of Cluster instances

    # Part B
    def getClusters(self):
        """
        Returns the list of clusters in this object.

        This method returns the cluster list directly (it does not copy).  Any changes
        made to this list will modify the set of clusters.
        """
        return self._cluster


    def __init__(self, dset, k, seeds=None):
        """
        Initializes the algorithm for the dataset ds, using k clusters.

        If the optional argument seeds is supplied, those seeds will be a list OR
        tuple of indices into the dataset. They specify which points should be the
        initial cluster centroids. Otherwise, the clusters are initialized by randomly
        selecting k different points from the database to be the cluster centroids.

        Parameter dset: the dataset
        Precondition: dset is an instance of Dataset

        Parameter k: the number of clusters
        Precondition: k is an int, 0 < k <= dset.getSize()

        Paramter seeds: the initial cluster indices (OPTIONAL)
        Precondition: seeds is None, or a list/tuple of valid seeds.
        """
        assert isinstance(dset, a6dataset.Dataset)
        assert type(k) == int
        assert 0 < k <= dset.getSize()
        if seeds is not None:
            assert valid_seeds(seeds, dset.getSize())

        self._dataset = dset

        result = []
        if seeds is None:
            nlist = random.sample(range(1, dset.getSize()), k)
            for x in nlist:
                result.append(a6cluster.Cluster(dset, dset.getPoint(x)))
        else:
            for x in seeds:
                result.append(a6cluster.Cluster(dset, dset.getPoint(x)))

        self._cluster = result


    def _nearest(self, point):
        """
        Returns the cluster nearest to point

        This method uses the distance method of each Cluster to compute the distance
        between point and the cluster centroid. It returns the Cluster that is closest.

        Ties are broken in favor of clusters occurring earlier in the list returned
        by getClusters().

        Parameter point: The point to compare.
        Precondition: point is a tuple of numbers (int or float). Its length is the
        same as the dataset dimension.
        """
        assert a6dataset.is_point(point) and len(point) == self._dataset._dimension

        min = self.getClusters()[0].distance(point)
        result = self.getClusters()[0]
        for x in range(len(self.getClusters())):
             dist = self.getClusters()[x].distance(point)
             if dist < min:
                 min = dist
                 result = self.getClusters()[x]
        return result


    def _partition(self):
        """
        Repartitions the dataset so each point is in exactly one Cluster.
        """
        for x in range(len(self._cluster)):
            self._cluster[x].clear()
        for x in range(len(self._dataset.getContents())):
            result = self._nearest(self._dataset.getContents()[x])
            result.addIndex(x)


    def _update(self):
        """
        Returns True if all centroids are unchanged after an update; False otherwise.

        This method first updates the centroids of all clusters'.  When it is done, it
        checks whether any of them have changed. It returns False if just one has
        changed. Otherwise, it returns True.
        """
        original = self._cluster
        result = True
        for x in range(len(self._cluster)):
            tmp = self._cluster[x].update()
            if tmp == False:
                result = False
        return result


    def step(self):
        """
        Returns True if the algorithm converges after one step; False otherwise.

        This method performs one cycle of the k-means algorithm. It then checks if
        the algorithm has converged and returns the appropriate result (True if
        converged, false otherwise).
        """
        self._partition()
        return self._update()


    def run(self, maxstep):
        """
        Continues clustering until either it converges or performs maxstep steps.

        After the maxstep call to step, if this calculation did not converge, this
        method will stop.

        Parameter maxstep: The maximum number of steps to perform
        Precondition: maxstep is an int >= 0
        """
        assert type(maxstep) == int
        assert maxstep >= 0

        for x in range(maxstep):
            if self.step():
                return self.step()
            else:
                self.step()
