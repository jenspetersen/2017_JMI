#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""

:AUTHOR: Jens Petersen
:ORGANIZATION: Heidelberg University Hospital; German Cancer Research Center
:CONTACT: jens.petersen@dkfz.de
:SINCE: Sat Nov 12 17:37:39 2016
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
import nibabel as nib
import numpy as np

# =============================================================================
# PROGRAM METADATA
# =============================================================================

__author__ = "Jens Petersen"
__email__ = "jens.petersen@dkfz.de"
__copyright__ = ""
__license__ = ""
__date__ = "Sat Nov 12 17:37:39 2016"
__version__ = "0.1"

# =============================================================================
# METHODS & CLASSES
# =============================================================================


def copy_from_file(path, name=None, dtype=None):
    """Load data into memory"""

    # if path is already a numpy array, just return
    if type(path) == np.ndarray:
        if dtype is not None:
            return path.astype(dtype)
        else:
            return path

    # use nibabel
    if path.endswith(".nii") or path.endswith(".nii.gz"):
        if dtype is None:
            data = nib.load(path).get_data()

    elif path.endswith(".h5") or path.endswith(".hdf5"):
        f = h5py.File(path)
        if name in f.keys():
            data = f[name][()]
        elif len(f.keys()) == 1:
            data = f[f.keys()[0]][()]
        else:
            raise IOError("None or too many datasets in file {}: {}.".format(
                path, f.keys()))

    else:
        raise IOError("Don't know how to handle file {}.".format(path))

    if dtype is None:
        return data
    else:
        return data.astype(dtype)


def save_h5(path, data, name="data", attributes={}, dtype=None):
    """Save data as H5 with attributes"""

    f = h5py.File(path)
    f.create_dataset(name, data=data, dtype=dtype)
    for attr in attributes:
        f[name].attrs[attr] = attributes[attr]
    f.close()


def makegroups_h5(groups, file_object):
    """Create group structure in HDF5 object"""

    if groups.startswith("/"):
        groups = groups[1:]

    if len(groups) == 0:
        return

    group_name = groups.split("/")[0]
    rest = "/".join(groups.split("/")[1:])

    if group_name not in file_object.keys():
        file_object.create_group(group_name)

    makegroups_h5(rest, file_object[group_name])


def copyattributes_h5(from_, to_):
    """Copy attributes from one HDF5 object to the other."""

    for key in from_.attrs.keys():
        to_.attrs[key] = from_.attrs[key]


def flat(arr):
    """Return arr flattened except for last axis."""

    shape = arr.shape[:-1]
    n_features = arr.shape[-1]

    return arr.reshape(np.product(shape), n_features)


def inflate(arr, shape):
    """Reshape arr to shape except for last axis."""

    target_shape = list(shape) + [arr.shape[-1]]
    return arr.reshape(*target_shape)


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
