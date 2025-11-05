#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  5 12:31:59 2025

@author: lenakemmelmeier
"""

#%% Set up environment - load in appropiate packages

import math
from scipy.stats import f as f_dist, ncf as ncf_dist

def partial_eta_squared_from_F(F, df1, df2):
    return (F*df1) / (F*df1 + df2)

def cohens_f_from_eta_p2(eta_p2):
    return math.sqrt(eta_p2 / (1 - eta_p2))

def power_from_F_effect(F, df1, df2, alpha=0.05):
    """
    Post-hoc power for an F-test using the noncentral F distribution.
    Works for ANOVA effects (incl. RM) given df1/df2 and observed F.
    """
    eta_p2 = partial_eta_squared_from_F(F, df1, df2)
    f2     = eta_p2 / (1 - eta_p2)        # Cohen's f^2
    lam    = f2 * df2                      # noncentrality parameter
    Fcrit  = f_dist.isf(alpha, df1, df2)   # upper-tail critical value
    power  = 1.0 - ncf_dist.cdf(Fcrit, df1, df2, lam)
    return eta_p2, math.sqrt(f2), lam, Fcrit, power

eta_p2, f, lam, Fcrit, power = power_from_F_effect(8.99, 1, 9, alpha=0.05)
print(f"partial eta^2 = {eta_p2:.3f}, Cohen's f = {f:.3f}, "
      f"lambda = {lam:.2f}, Fcrit = {Fcrit:.3f}, power ≈ {power:.2f}")




