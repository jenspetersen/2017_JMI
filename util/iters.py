#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""

:AUTHOR: Jens Petersen
:ORGANIZATION: Heidelberg University Hospital; German Cancer Research Center
:CONTACT: jens.petersen@dkfz.de
:SINCE: Wed Nov  9 17:58:37 2016
:VERSION: 0.1

DESCRIPTION
-----------



REQUIRES
--------



TODO
----



"""
# =============================================================================
# IMPORT STATEMENTS
# =============================================================================

import h5py
import itertools as it

# =============================================================================
# PROGRAM METADATA
# =============================================================================

__author__ = "Jens Petersen"
__email__ = "jens.petersen@dkfz.de"
__copyright__ = ""
__license__ = ""
__date__ = "Wed Nov  9 17:58:37 2016"
__version__ = "0.1"

# =============================================================================
# METHODS & CLASSES
# =============================================================================


def kwargs(parameters, *args):
    """For a dictionary of parameter lists, yield possible combinations.

    Combinations will be returned as dictionaries. Additional arguments
    will no be split (lists remain lists etc.)"""

    i = 0

    # make everything iterable
    # copy values to not change input object
    keys = sorted(parameters.keys())
    values = []

    for key in keys:
        if hasattr(parameters[key], "__iter__") and key not in args:
            values.append(parameters[key])
        else:
            values.append([parameters[key]])

    for comb in it.product(*values):
        yield i, dict(zip(keys, comb))
        i += 1


def walk_h5(h5object, yield_datasets=True):
    """Similar to os.walk, walk through tree structure in a HDF5 file.

    Yields (current object, [dataset names], [group names])."""

    if isinstance(h5object, h5py.Dataset):

        if yield_datasets:
            yield h5object, [], []
        else:
            raise TypeError("Datasets ignored but object ist dataset.")

    else:

        groups = []
        datasets = []

        for key in h5object.keys():
            if isinstance(h5object[key], h5py.Group):
                groups.append(key)
            if isinstance(h5object[key], h5py.Dataset):
                datasets.append(key)

        yield h5object, groups, datasets

        for g in groups:
            for el in walk_h5(h5object[g], yield_datasets):
                yield el

        if yield_datasets:
            for d in datasets:
                for el in walk_h5(h5object[d], yield_datasets):
                    yield el


# =============================================================================
# MAIN METHOD
# =============================================================================


def main():

    import IPython
    IPython.embed()

# =============================================================================
# RUN
# =============================================================================

if __name__ == "__main__":

    main()
