"""
Dataset for k-Means clustering

This file contains the class Dataset, which is the very first part of the assignment.
You cannot do anything in this assignment (except run the unit test) before this class
is finished.

Akhil Damani ad674
11/17/21
"""
import math
import random
import numpy


# HELPERS TO CHECK PRECONDITIONS
def is_point(value):
    """
    Returns True if value is a tuple of int or float

    Parameter value: a value to check
    Precondition: value can be anything
    """
    if (type(value) != tuple):
        return False

    # All float
    okay = True
    for x in value:
        if (not type(x) in [int,float]):
            okay = False

    return okay


def is_point_list(value):
    """
    Returns True if value is a list of points (int/float tuples)

    This function also checks that all points in value have same dimension.

    Parameter value: a value to check
    Precondition: value can be anything
    """
    if value == []:
        return True
    if type(value) == list:
        for x in range(len(value)):
            if is_point(value[x]) and len(value[0]) == len(value[x]):
                result = True
            else:
                return False
    else:
        result = False

    return result


class Dataset(object):
    """
    A class representing a dataset for k-means clustering.

    The data is stored as a list of points (int/float tuples). All points have
    the same number elements which is the dimension of the data set.

    None of the attributes should be accessed directly outside of the class Dataset
    (e.g. in the methods of class Cluster or KMeans). Instead, this class has getter and
    setter style methods (with the appropriate preconditions) for modifying these values.
    """
    # IMMUTABLE ATTRIBUTES (Fixed after initialization)
    # Attribute _dimension: The point dimension for this dataset
    # Invariant: _dimension is an int > 0.
    #
    # MUTABLE ATTRIBUTES (Can be changed at any time, via addPoint)
    # Attribute _contents:  The dataset contents
    # Invariant: _contents is a list of tuples of numbers (float or int), possibly empty.
    # Each element of _contents is a tuple of size _dimension

    # Getters for encapsulated attributes
    def getDimension(self):
        """
        Returns the point dimension of this data set
        """
        return self._dimension


    def getSize(self):
        """
        Returns the number of points in this data set.
        """
        if self._contents is None:
            return 0
        else:
            return len(self._contents)


    def getContents(self):
        """
        Returns the contents of this data set as a list of points.

        This method returns the contents directly (not a copy). Any changes made to this
        list will modify the data set. If you want to access the data set, but want to
        protect yourself from modifying the data, use getPoint() instead.
        """
        return self._contents


    def __init__(self, dim, contents=None):
        """
        Initializes a database for the given point dimension.

        The optional parameter contents is the initial value of the of the data set.
        When intializing the data set, it creates a copy of the list contents.
        However, since tuples are not mutable, it does not need to copy the points
        themselves.  Hence a shallow copy is acceptable.

        If contents is None, the data set start off empty. The parameter contents is
        None by default.

        Parameter dim: The dimension of the dataset
        Precondition: dim is an int > 0

        Parameter contents: the dataset contents
        Precondition: contents is either None or a list of points (int/float tuples).
        If contents is not None, then contents is not empty and the length of each
        point is equal to dim.
        """
        assert type(dim) == int and dim > 0
        assert contents is None or is_point_list(contents)

        self._dimension = dim
        if contents is None:
            self._contents = []
        else:
            self._contents = contents[:]


    def getPoint(self, i):
        """
        Returns the point at index i in this data set.

        Often, we want to access a point in the data set, but we want to make sure that
        we do not accidentally modify the data set.  That is the purpose of this method.
        Since tuples are not mutable, giving access to the tuple without access to the
        underlying list is safer.

        If you actually want to modify the data set, use the method getContents().
        That returns the list storing the data set, and any changes to that list will
        alter the data set.

        Parameter i: the index position of the point
        Precondition: i is an int that refers to a valid position in 0..getSize()-1
        """
        assert type(i) == int
        assert 0 <= i <= self.getSize()-1

        return self._contents[i]


    def addPoint(self,point):
        """
        Adds a the given point to the end of this data set.

        The point does not need to be copied since tuples are not mutable.

        Parameter point: The point to add to the set
        Precondition: point is a tuple of int/float. The length of point is equal
        to getDimension().
        """
        assert is_point(point)
        assert len(point) == self.getDimension()

        self._contents.append(point)
