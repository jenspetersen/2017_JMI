#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""

:AUTHOR: Jens Petersen
:ORGANIZATION: Heidelberg University Hospital; German Cancer Research Center
:CONTACT: jens.petersen@dkfz.de
:SINCE: Thu Nov 10 15:53:33 2016
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

import fnmatch
import inspect
import numpy as np
import os
import re

# =============================================================================
# PROGRAM METADATA
# =============================================================================

__author__ = "Jens Petersen"
__email__ = "jens.petersen@dkfz.de"
__copyright__ = ""
__license__ = ""
__date__ = "Thu Nov 10 15:53:33 2016"
__version__ = "0.1"

# =============================================================================
# METHODS & CLASSES
# =============================================================================


class Bunch:
    """Convenience class to represent dictionary.

    dict[key] -> Bunch.key"""

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class VerboseConsole:

    def __init__(self, verbose):
        self.verbose = verbose

    def out(self, string):
        if self.verbose:
            print string


def to_num(string):
    """Convert string to number (or None) if possible"""

    if type(string) != str:
        return string

    if string == "None":
        return None

    if re.match("\d+\.\d*$", string):
        return float(string)
    elif re.match("\d+$", string):
        return int(string)
    else:
        return string


def parameter_dict(parsed_list):
    """Convert a list of keyword arguments to a dictionary"""

    if type(parsed_list) == str:
        parsed_list = parsed_list.split(" ")

    param = {}

    for el in parsed_list:
        key, val = el.split("=")
        if val.startswith("["):
            val = val[1:-1].replace(" ", "").split(",")
            val = map(to_num, val)
        elif val.startswith("("):
            val = val[1:-1].replace(" ", "").split(",")
            val = map(to_num, val)
            val = tuple(val)
        else:
            val = to_num(val)
        param[key] = val

    return param


def numeric_list_from_string(parsed_string):
    """Make a list from a string."""

    if parsed_string is None:
        return None

    if parsed_string.startswith("[") or parsed_string.startswith("("):
        parsed_string = parsed_string[1:]
    if parsed_string.endswith("]") or parsed_string.endswith(")"):
        parsed_string = parsed_string[:-1]

    list_ = parsed_string.split(" ")
    if len(list_) == 1:
        list_ = parsed_string.split(",")

    for i in range(len(list_)):
        if list_[i].startswith("[") or list_[i].startswith("("):
            list_[i] = numeric_list_from_string(list_[i])
        else:
            list_[i] = to_num(list_[i])

    return list_


def numeric_list_from_list(parsed_list):
    """Make list numeric."""

    if parsed_list is None:
        return None

    new_list = []

    for el in parsed_list:
        if el.startswith("[") or el.startswith("("):
            new_list.append(numeric_list_from_string(el))
        else:
            new_list.append(to_num(el))

    return new_list


def pattern_path(base_dir, pattern, unique=True):
    """Find paths within base_dir that match pattern."""

    full_pattern = os.path.join(base_dir, pattern)
    matches = []

    for dirpath, dirnames, filenames in os.walk(base_dir):
        if fnmatch.fnmatch(dirpath, full_pattern):
            matches.append(dirpath)
        for file_ in filenames:
            if fnmatch.fnmatch(os.path.join(dirpath, file_), full_pattern):
                matches.append(os.path.join(dirpath, file_))

    if not unique:
        return matches

    else:
        if len(matches) != 1:
            raise RuntimeError(
                "Could not find a unique match for {} in {}".format(
                    pattern, base_dir))
        else:
            return matches[0]


def id_from_list(pattern, list_, require_unique=True):
    """Find all occurrences of pattern in list of strings."""

    result = []

    for el in list_:
        match = re.search(pattern, el)
        if match:
            result.append(match.group(0))

    if require_unique:
        if len(result) != len(set(result)):
            raise RuntimeError("Identifiers not unique.")

    return result


def none_string(el):
    """Return input or 'None' if None"""

    if el is None:
        return "None"
    else:
        return el


def config_for_function(func, config):
    """Return a subset of config that applies to func."""

    func_config = {}
    for arg in inspect.getargspec(func).args:
        if arg in config.keys():
            func_config[arg] = config[arg]
    return func_config


def extend_dict(dict1, dict2):
    """Merge dict entries to lists."""

    for key in dict2:
        if key not in dict1:
            dict1[key] = dict2[key]
        else:
            if not hasattr(dict1[key], "__iter__"):
                if not hasattr(dict2[key], "__iter__"):
                    if dict1[key] != dict2[key]:
                        dict1[key] = [dict1[key], dict2[key]]
                else:
                    if dict1[key] in dict2[key]:
                        dict1[key] = dict2[key]
                    else:
                        dict1[key] = [dict1[key]]
                        dict1[key].extend(dict2[key])
            else:
                if not hasattr(dict2[key], "__iter__"):
                    if dict2[key] not in dict1[key]:
                        dict1[key].append(dict2[key])
                else:
                    dict1[key] = list(np.union1d(dict1[key], dict2[key]))

    return dict1

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
