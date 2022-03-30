"""
Useful tools k-means clustering

This module is where we put the helpers that are used by multiple other modules.

Authors: Walker M. White (wmw2)
Date: October 26, 2021
"""
import traceback
import introcs
import os, sys


def data_for_file(filename):
    """
    Returns the 2-dimensional table for the given CSV file.
    
    CSV files should have a header with attributes.  The header COMMENTS is ignored.
    All other attributes are kept and should have numeric values.
    
    Parameter filename: The file to parse
    Precondition: filename is a name of a CSV file.
    """
    try:
        if filename is None:
            return None
        
        table = introcs.read_csv(filename)
        
        # Is there a column called COMMENTS?
        header = list(map(lambda x: x.lower(), table[0]))
        pos = header.index('comments') if 'comments' in header else len(header)
        contents = []
        for row in table[1:]:
            point = row[:pos]+row[pos+1:]
            contents.append(tuple(map(float,point)))
        
        return contents
    except:
        traceback.print_exc()
        raise AssertionError('%s is not a valid dataset' % repr(filename))


def list_csv(directory,suffix=None):
    """
    Returns the list of CSV files in a directory.

    The optional suffix attribute is used to separate 2d CSV files from other,
    more general files.

    Parameter directory: The directory path
    Precondition: directory is a string and valid path name

    Parameter suffix: The suffix BEFORE the .csv extension.
    Precondition: suffix is a string
    """
    result = []
    for item in os.listdir(directory):
        pair = os.path.splitext(item)
        if pair[1] == '.csv':
            if not suffix:
                result.append(pair[0])
            elif pair[0].endswith(suffix):
                result.append(pair[0][:-len(suffix)])
    result.sort()
    return result


def compute(filename,k,limit=500):
    """
    Computes the result of a k-means algorithm and returns it as a table [with header]
    
    Parameter filename: The name of the initial dataset
    Precondition: filename is a valid file path OR None.
    
    Parameter k: The initial number of clusters
    Precondition: k is an int
    
    Parameter limit: The limit on the number of iterations to run
    Precondition: limit is an int >= 0
    """
    data = data_for_file(filename)
    if len(data) == 0:
        return []
    
    # Get the header again
    table = introcs.read_csv(filename)
    # Is there a column called COMMENTS?
    header = list(filter(lambda x: x.lower() != 'comments', table[0]))
    
    import a6dataset
    import a6algorithm
    dset = a6dataset.Dataset(len(data[0]), data)
    km   = a6algorithm.Algorithm(dset, k)
    km.run(limit)
    
    result = []
    newhead = ['CID']+header
    for x in header:
        newhead.append('C_'+x)
    result.append(newhead)
    
    clusters = km.getClusters()
    for x in range(len(clusters)):
        for y in clusters[x].getIndices():
            row = [x+1]+clusters[x]._dataset.getPoint(y) # BAD
            row.extend(clusters[x].getCentroid())
            result.append(row)
    
    return result
