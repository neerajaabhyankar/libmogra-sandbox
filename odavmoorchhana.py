# Author: Neeraja Abhyankar
# Created: 22 May 2021

# %% IMPORTS

import os
import time
import numpy as np

# %% PROBLEM DEFINITION

"""
I want to write a program that characterizes moorchhana-groups of ragas completely.

"""

# %% DATA STRUCTURES

# saptak

swar = [
    "sa",
    "komal-re",
    "re",
    "komal-ga",
    "ga",
    "ma",
    "teevra-ma",
]
chakra_5ths = ["sa", "pa", "re", "ma"]
# TODO: Make this data structure cyclic

# ODAV raag niyam aani ODAV raag

forbidden_pairs = [
    ("komal-re", "re"),
]

# moorchhana operations

# TODO: "moorgat" = a list of "cycles";
# "cycle" = a cyclic set of 5 raags, some valid some invalid, with one primal raag, lexically determined.

# %%
