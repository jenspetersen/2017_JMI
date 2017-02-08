#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""

:AUTHOR: Jens Petersen
:ORGANIZATION: Heidelberg University Hospital; German Cancer Research Center
:CONTACT: jens.petersen@dkfz.de
:SINCE: Wed Nov 16 17:53:24 2016
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

import metrics
import numpy as np
import random
from skimage.measure import label as sklabel
from skimage.morphology import opening

# =============================================================================
# PROGRAM METADATA
# =============================================================================

__author__ = "Jens Petersen"
__email__ = "jens.petersen@dkfz.de"
__copyright__ = ""
__license__ = ""
__date__ = "Wed Nov 16 17:53:24 2016"
__version__ = "0.1"

# =============================================================================
# METHODS & CLASSES
# =============================================================================


def training_data_from_indices(data, groundtruth, indices):
    """Construct training data from indices. Feature axis = -1"""

    X = []
    y = []
    for index in indices:
        X.append(data[index])
        y.append(groundtruth[index])
    return np.asarray(X, dtype=np.float32), np.asarray(y, dtype=np.int)


def random_indices(shape, number):
    """Draw n = number len(shape)-dimensional indices"""

    indices = []
    while len(indices) < number:
        index = map(np.random.randint, shape)
        indices.append(tuple(index))
    return indices


def random_stroke(shape,
                  strokelength,
                  mask=None,
                  failure_threshold=1000,
                  groundtruth=None):
    """Line of length strokelength along random axis with random starting point
    in shape, from masked region if desired. If groundtruth given, strokes will
    not cross classes."""

    if strokelength > np.max(shape):
        raise IndexError("Strokelength too big for array")
    if groundtruth is not None and groundtruth.shape != shape:
        raise IndexError("Groundtruth shape does not match given shape")

    failure_count = 0

    if mask is None:

        while failure_count < failure_threshold:

            start_index = tuple(map(np.random.randint, shape))
            d = np.random.randint(len(shape))

            if start_index[d] + strokelength - 1 >= shape[d]:

                failure_count += 1
                continue

            else:

                indices = []
                for i in range(strokelength):
                    index = list(start_index)
                    index[d] += i
                    indices.append(tuple(index))

                if groundtruth is not None:
                    classes = set()
                    for index in indices:
                        classes.add(groundtruth[index])
                    if len(classes) > 1:
                        failure_count += 1
                        continue

                return indices

    else:

        mask_indices = np.where(mask)

        while failure_count < failure_threshold:

            start_index = random.choice(zip(*mask_indices))
            d = np.random.randint(len(shape))
            end_index = list(start_index)
            end_index[d] += strokelength - 1

            if end_index[d] >= shape[d]:

                failure_count += 1
                continue

            else:

                indices = []
                for i in range(strokelength):
                    index = list(start_index)
                    index[d] += i
                    indices.append(tuple(index))

                in_mask = True
                for index in indices:
                    if mask[index] == 0:
                        in_mask = False
                if not in_mask:
                    failure_count += 1
                    continue

                if groundtruth is not None:
                    classes = set()
                    for index in indices:
                        classes.add(groundtruth[index])
                    if len(classes) > 1:
                        failure_count += 1
                        continue

                return indices

    raise IndexError("Could not find stroke in array")


def draw_indices(shape,
                 stroke_length,
                 annotation_mode,
                 groundtruth=None,
                 prediction=None,
                 uncertainty_mode="entropy",
                 uncertainty_threshold=0.8):
    """Draw random strokes for different annotation modes."""

    if annotation_mode == "random":

        return random_stroke(shape,
                             stroke_length,
                             failure_threshold=1000,
                             groundtruth=groundtruth)

    elif annotation_mode == "uncertainty":

        assert prediction is not None

        # create mask where uncertainty is above threshold percentage
        # we do not check whether uncertainty_mode really exists
        mask = eval("metrics.{}(prediction)".format(uncertainty_mode))
        min_ = np.min(mask)
        max_ = np.max(mask)
        mask = mask > (uncertainty_threshold * (max_ - min_) + min_)

        try:
            return random_stroke(shape,
                                 stroke_length,
                                 mask=mask,
                                 failure_threshold=1000,
                                 groundtruth=groundtruth)
        except IndexError:
            return random_stroke(shape,
                                 stroke_length,
                                 failure_threshold=1000,
                                 groundtruth=groundtruth)

    elif annotation_mode == "corrective":

        assert groundtruth is not None
        assert prediction is not None

        # create current segmentation and select random label from it
        seg = np.argmax(prediction, axis=-1)
        labels = np.unique(seg)
        l = random.choice(labels)

        # find region where we are wrong
        mask = (seg == l) * (groundtruth != l)

        try:
            return random_stroke(shape,
                                 stroke_length,
                                 mask=mask,
                                 failure_threshold=1000,
                                 groundtruth=groundtruth)
        except IndexError:
            return random_stroke(shape,
                                 stroke_length,
                                 failure_threshold=1000,
                                 groundtruth=groundtruth)

    else:

        raise ValueError("Unknown annotation mode.")


def weighted_start_indices(truth, number, labels=None):

    if number == 0:
        return []

    if labels is None:
        labels = np.unique(truth)

    if type(number) == int:

        weights = map(lambda x: np.mean(truth == x), labels)
        counts = map(lambda x: int(number * x), weights)

        for i in range(len(counts)):
            if counts[i] == 0:
                counts[i] = 1
                counts[np.argmax(counts)] -= 1
        while np.sum(counts) > number:
            counts[np.argmax(counts)] -= 1
        while np.sum(counts) < number:
            counts[np.random.randint(len(counts))] += 1

    else:

        counts = number

    indices_all = []

    for i, label in enumerate(labels):
        indices_current = []
        indices = zip(*np.where(truth == label))
        while len(indices_current) < counts[i] and len(indices) > 0:
            index = random.choice(indices)
            indices_current.append(index)
            indices.remove(index)
        indices_all += indices_current

    return indices_all


def consensus(arr, axis=0):
    """Find consensus in array along axis. Returns binary array."""

    target_shape = list(arr.shape)
    del target_shape[axis]
    committee_size = arr.shape[axis]

    consensus = np.ones(target_shape, dtype=np.bool)

    for i in xrange(committee_size - 1):
        for j in xrange(i+1, committee_size):
            slc1 = [slice(None)] * arr.ndim
            slc2 = [slice(None)] * arr.ndim
            slc1[axis] = i
            slc2[axis] = j
            consensus *= arr[slc1] == arr[slc2]

    return consensus


def maximum_uncertainty_region(data,
                               uncertainty_threshold="mean",
                               morph_opening_struct=None,
                               region_mode="sum"):
    """Return mask where data has the largest connected region > threshold"""

    if np.min(data) == np.max(data):
        return np.ones(data.shape, dtype=np.float32)

    # set threshold to mean if nothing was specified
    if uncertainty_threshold == "mean":
        uncertainty_threshold = np.mean(data)
    else:
        uncertainty_threshold = uncertainty_threshold * (np.max(data) - np.min(data)) + np.min(data)


    thresh_data = data >= uncertainty_threshold
    if morph_opening_struct is not None:
        thresh_data = opening(thresh_data, morph_opening_struct)

    # create thresholded image and image with labeled components
    regions_labeled = sklabel(thresh_data, background=0)
    labels = np.unique(regions_labeled)

    # get uncertainty for each region
    uncertainties = [0]
    for label in labels:
        if label == 0:
            continue
        if region_mode == "sum":
            uncertainty = np.sum(data * (regions_labeled == label))
        elif region_mode in ("average", "mean"):
            uncertainty = np.sum(data * (regions_labeled == label)) / np.float(np.sum(regions_labeled == label))
        else:
            raise ValueError("Unknown option for region_mode: {}".format(region_mode))
        uncertainties.append(uncertainty)
    max_component = np.argmax(uncertainties)

    # create mask of target region
    return regions_labeled == max_component


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
