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

cv = mtb.cross_validation.LeaveOneOut(random_state=42)

X_train, X_test, y_train, y_test = mtb.cross_validation.train_test_split(cv,X,y)

np.random.seed(12)
groups = np.random.randint(1,3,10)
cv = mtb.cross_validation.LeavePSubGroupOut(random_state=42)

X_train, X_test, y_train, y_test = mtb.cross_validation.train_test_split(cv,X,y,random_state=42,groups=groups)