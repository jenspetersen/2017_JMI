#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""

:AUTHOR: Jens Petersen
:ORGANIZATION: Heidelberg University Hospital; German Cancer Research Center
:CONTACT: jens.petersen@dkfz.de
:SINCE: Tue Nov  8 16:50:16 2016
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

import collections
import inspect
import numpy as np
import pandas as pd

# =============================================================================
# PROGRAM METADATA
# =============================================================================

__author__ = "Jens Petersen"
__email__ = "jens.petersen@dkfz.de"
__copyright__ = ""
__license__ = ""
__date__ = "Tue Nov  8 16:50:16 2016"
__version__ = "0.1"

# =============================================================================
# METHODS & CLASSES
# =============================================================================


class EvaluationSuite:

    _metrics = ["dice", "jaccard", "precision", "recall"]

    def __init__(self,
                 test=None,
                 reference=None,
                 labels=None,
                 label_names=None):

        self.test = None
        self.reference = None
        self.labels = None
        self.label_names = None
        self.result = None
        self.metrics = []
        for m in self._metrics:
            self.metrics.append(m)

        self.set_test(test)
        self.set_reference(reference)
        self.set_labels(labels)
        self.set_label_names(label_names)

    def set_test(self, test):
        """Set the test segmentation."""

        self.test = test

    def set_reference(self, reference):
        """Set the reference segmentation."""

        self.reference = reference

    def set_labels(self, labels):
        """Set the labels."""

        self.labels = labels
        if labels is not None:
            for i in range(len(labels)):
                if not isinstance(labels[i], collections.Hashable):
                    self.labels[i] = tuple(labels[i])

    def set_label_names(self, label_names):
        """Set the label names."""

        self.label_names = label_names

    def construct_labels(self):
        """Construct label dictionary from unique entries in segmentations."""

        if self.test is None and self.reference is None:
            raise ValueError("No test or reference segmentations.")
        elif self.test is None:
            self.labels = np.unique(self.reference)
        elif self.reference is None:
            self.labels = np.unique(self.test)
        else:
            self.labels = np.union1d(np.unique(self.test),
                                     np.unique(self.reference))

    def set_metrics(self, metrics):
        """Set evaluation metrics"""

        self.metrics = metrics

    def evaluate(self):
        """Compute metrics for segmentations."""

        if self.test is None or self.reference is None:
            raise ValueError("Need both test and reference segmentations.")

        assert self.test.shape == self.reference.shape,\
            "Shape mismatch: {} and {}".format(self.test.shape,
                                               self.reference.shape)

        if self.labels is None:
            self.construct_labels()
        if self.label_names is not None:
            assert len(self.labels) == len(self.label_names),\
                "Number of label names does not match number of labels."

        # get functions for evaluation
        # somewhat convoluted, but allows users to define additonal metrics
        # on the fly, e.g. inside an IPython console
        _funcs = {m: eval(m) for m in self.metrics}
        frames = inspect.getouterframes(inspect.currentframe())
        for m in self.metrics:
            for f in frames:
                if m in f[0].f_locals:
                    _funcs[m] = f[0].f_locals[m]
                    break
            else:
                if m in _funcs:
                    continue
                else:
                    raise NotImplementedError(
                        "Metric {} not implemented.".format(m))

        # get results
        self.result = {}

        for i, l in enumerate(self.labels):

            self.result[l] = {}

            for m in self.metrics:

                if not hasattr(l, "__iter__"):

                    self.result[l][m] = _funcs[m](
                        self.test == l,
                        self.reference == l)

                else:

                    current_test = 0
                    current_reference = 0
                    for label in l:
                        current_test += (self.test == label)
                        current_reference += (self.reference == label)

                    self.result[l][m] = _funcs[m](current_test,
                                                  current_reference)
        if self.label_names is not None:
            for i, name in enumerate(self.label_names):
                self.result[name] = self.result[self.labels[i]]

        return self.result

    def to_array(self):
        """Return result as numpy array (labels x metrics)."""

        if self.result is None:
            self.evaluate()

        assert sorted(self.metrics) ==\
            sorted(self.result[self.result.keys()[0]]),\
            "Metrics in last result do not match object metrics. Array \
            construction is ambiguous."

        a = np.zeros((len(self.labels), len(self.metrics)), dtype=np.float32)

        for i, l in enumerate(self.labels):
            for j, m in enumerate(self.metrics):
                a[i][j] = self.result[l][m]

        return a

    def to_pandas(self):
        """Return result as pandas DataFrame."""

        a = self.to_array()

        if self.label_names is not None:
            assert len(self.labels) == len(self.label_names)
            labels = self.label_names
        else:
            labels = self.labels

        return pd.DataFrame(a, index=labels, columns=self.metrics)


def dice(test, reference):
    """Calculate Soerensen-Dice coefficient for two segmentations.

    Segmentations will be treated as binary with non-zero values counting
    towards the segmentation. Result is symmetric with respect to test and
    reference distinction."""

    assert type(test) == np.ndarray, "Test type: {}".format(type(test))
    assert type(reference) == np.ndarray,\
        "Reference type: {}".format(type(reference))
    assert test.shape == reference.shape, "Shapes {} and {}".format(
        test.shape, reference.shape)
    if not (np.any(test) and np.any(reference)):
        return 0.

    return 2. * np.sum((test != 0)*(reference != 0)) /\
        (np.sum(test != 0, dtype=np.float32) +
         np.sum(reference != 0, dtype=np.float32))


def jaccard(test, reference):
    """Calculate Jaccard coefficient for two segmentations.

    Segmentations will be treated as binary with non-zero values counting
    towards the segmentation. Result is symmetric with respect to test and
    reference distinction."""

    assert type(test) == np.ndarray, "Test type: {}".format(type(test))
    assert type(reference) == np.ndarray,\
        "Reference type: {}".format(type(reference))
    assert test.shape == reference.shape, "Shapes {} and {}".format(
        test.shape, reference.shape)
    if not (np.any(test) and np.any(reference)):
        return 0.

    return np.sum((test != 0)*(reference != 0)) /\
        np.sum((test + reference) != 0, dtype=np.float32)


def precision(test, reference):
    """Calculate precision of test segmentation with respect to reference.

    Segmentations will be treated as binary with non-zero values counting
    towards the segmentation."""

    assert type(test) == np.ndarray, "Test type: {}".format(type(test))
    assert type(reference) == np.ndarray,\
        "Reference type: {}".format(type(reference))
    assert test.shape == reference.shape, "Shapes {} and {}".format(
        test.shape, reference.shape)
    if not (np.any(test) and np.any(reference)):
        return 0.

    return np.sum((test != 0)*(reference != 0)) /\
        np.sum(test != 0, dtype=np.float32)


def recall(test, reference):
    """Calculate recall of test segmentation with respect to reference.

    Segmentations will be treated as binary with non-zero values counting
    towards the segmentation."""

    assert type(test) == np.ndarray, "Test type: {}".format(type(test))
    assert type(reference) == np.ndarray,\
        "Reference type: {}".format(type(reference))
    assert test.shape == reference.shape, "Shapes {} and {}".format(
        test.shape, reference.shape)
    if not (np.any(test) and np.any(reference)):
        return 0.

    return np.sum((test != 0)*(reference != 0)) /\
        np.sum(reference != 0, dtype=np.float32)


def sensitivity(test, reference):
    """Calculate sensitivity of test segmentation with respect to reference.

    Segmentations will be treated as binary with non-zero values counting
    towards the segmentation."""

    return recall(test, reference)


def specificity(test, reference):
    """Calculate specificity of test segmentation with respect to reference.

    Segmentations will be treated as binary with non-zero values counting
    towards the segmentation."""

    assert type(test) == np.ndarray, "Test type: {}".format(type(test))
    assert type(reference) == np.ndarray,\
        "Reference type: {}".format(type(reference))
    assert test.shape == reference.shape, "Shapes {} and {}".format(
        test.shape, reference.shape)
    if not (np.any(test) and np.any(reference)):
        return 0.

    return np.sum((test == 0)*(reference == 0)) /\
        np.sum(reference == 0, dtype=np.float32)


def accuracy(test, reference):
    """Calculate accuracy of test segmentation with respect to reference.

    Segmentations will be treated as binary with non-zero values counting
    towards the segmentation."""

    assert type(test) == np.ndarray, "Test type: {}".format(type(test))
    assert type(reference) == np.ndarray,\
        "Reference type: {}".format(type(reference))
    assert test.shape == reference.shape, "Shapes {} and {}".format(
        test.shape, reference.shape)
    if not (np.any(test) and np.any(reference)):
        return 0.

    return (np.sum((test == 0)*(reference == 0)) +
            np.sum((test != 0)*(reference != 0))) /\
        np.product(reference.shape, dtype=np.float32)


def fscore(test, reference, beta=1.):
    """Calculate F-score of test segmentation with respect to reference.

    Segmentation will be treated as binary with non-zero values counting
    towards the segmentation. For the meaning of beta please see
    https://en.wikipedia.org/wiki/F1_score."""

    assert type(test) == np.ndarray
    assert type(reference) == np.ndarray
    assert test.shape == reference.shape

    precision_ = precision(test, reference)
    recall_ = recall(test, reference)

    return (1 + beta*beta) * precision_ * recall_ /\
        ((beta*beta * precision_) + recall_)


def entropy(probs, axis=-1, eps=0.000001):
    """Calculate entropy of probabilities along axis"""

    assert type(probs) == np.ndarray

    probs = np.array(probs, dtype=np.float32)
    N = probs.shape[axis]
    shape = list(probs.shape)
    del shape[axis]

    if N == 1:
        raise ValueError("Cannot calculate entropy for single values.")

    else:
        sum_ = np.sum(probs, axis)
        if np.any(sum_ != 1):
            probs /= np.expand_dims(sum_, axis)
        probs[probs == 0] = eps
        return - np.sum(probs * np.log(probs), axis) / np.log(N)


def inv_margin(probs, axis=-1):
    """Calculate 1 - probability margin along axis"""

    assert type(probs) == np.ndarray

    probs = np.array(probs, dtype=np.float32)
    N = probs.shape[axis]
    shape = list(probs.shape)
    del shape[axis]

    if N == 1:
        raise ValueError("Cannot calculate margin for single values.")

    else:
        sum_ = np.sum(probs, axis)
        if np.any(sum_ != 1):
            probs /= np.expand_dims(sum_, axis)
        return 1. - (np.max(probs, axis) -
                     np.split(np.partition(probs, N-2, axis),
                              N, axis)[1].reshape(*shape))


def inv_confidence(probs, axis=-1):
    """Calculate 1 - confidence with respect to probabilities along axis"""

    assert type(probs) == np.ndarray

    probs = np.array(probs, dtype=np.float32)
    N = probs.shape[axis]
    shape = list(probs.shape)
    del shape[axis]

    if N == 1:
        return np.zeros(shape, dtype=np.float32)

    else:
        sum_ = np.sum(probs, axis)
        if np.any(sum_ != 1):
            probs /= np.expand_dims(sum_, axis)
        return (1. - np.max(probs, axis)) / (1. - 1./N)

# =============================================================================
# MAIN METHOD
# =============================================================================


def main():

    # for testing
    import IPython
    IPython.embed()

# =============================================================================
# RUN
# =============================================================================

if __name__ == "__main__":

    main()
