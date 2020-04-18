#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train test split with every kind of cross-validation
======================================================

This example shows how to split between test and train according to
every cross-validation method.
"""

##############################################################################
# Import librairies
# -------------------------------------------

import numpy as np
import museotoolbox as mtb

##############################################################################
# Generate random dataset
# -------------------------------------------

np.random.seed(42)
y = np.random.randint(1,3,10)
X = np.random.randint(1,255,[10,3],dtype=np.uint8)

##############################################################################
# Split train/test 
# -----------------------------------------------------------------------
# Using :mod:`museotoolbox.cross_validation.LeaveOneOut`

cv = mtb.cross_validation.LeaveOneOut(random_state=42)

X_train, X_test, y_train, y_test = mtb.cross_validation.train_test_split(cv,X,y)

##############################################################################
# Split train/test with groups
# -------------------------------------------
# Generate group

groups = np.array([1, 1, 2, 3, 4, 2, 1, 1, 2, 3],dtype=int)

##################################################################
# Using :mod:`museotoolbox.cross_validation.LeaveOneSubGroupOut`

cv = mtb.cross_validation.LeaveOneSubGroupOut(random_state=42)

X_train, X_test, y_train, y_test = mtb.cross_validation.train_test_split(cv,X,y,groups=groups)