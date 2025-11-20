#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  5 12:31:59 2025

@author: lenakemmelmeier
"""


#%% set up environment - load in appropiate packages

import math
from scipy.stats import f as f_dist, ncf as ncf_dist

#%% helpers (effect sizes, posthoc power, sample size estimation)

# got this formulua from google (getting partial eta)
def effect_sizes_from_F(F, df1, df2):
    eta_p2 = (F * df1) / (F * df1 + df2)
    f = math.sqrt(eta_p2 / (1.0 - eta_p2)) # cast as a float (Cohen's f)
    
    return eta_p2, f

# it looks like statmodels actually isn't great at power for N-way ANOVAs
# calculate posthoc power manually using the noncentral F
def posthoc_power_from_F(F, df1, df2, alpha=0.05):
    eta_p2, f = effect_sizes_from_F(F, df1, df2) # get partial eta^2 and Cohen's f
    lam = (f * f) * df2 # noncentrality parameter (lambda = f^2 * df2)
    Fcrit = f_dist.isf(alpha, df1, df2) # get critical F value (upper-tail cutoff)
    power = 1.0 - ncf_dist.cdf(Fcrit, df1, df2, lam) # probability that F > Fcrit
    
    return eta_p2, f, power

# compute power for a given N (using observed Cohen's f)
def power_from_f_N(f, df1, N, alpha=0.05, df2_from_N=lambda N: N - 1):
    df2 = df2_from_N(N)
    lam = (f * f) * df2
    Fcrit = f_dist.isf(alpha, df1, df2)
    power = 1.0 - ncf_dist.cdf(Fcrit, df1, df2, lam)
    
    return power

# finding smallest N reaching a given target power (no rounding!)
def N_for_target_power(f, df1, target_power, alpha=0.05, start_N=4, max_N=2000, df2_from_N=lambda N: N - 1):
    N = max(start_N, 2)
   
    while N <= max_N:
        pwr = power_from_f_N(f, df1, N, alpha=alpha, df2_from_N=df2_from_N)
        
        if pwr >= target_power:
            return N, pwr
        
        N += 1
#%% using reported values from paper

eta_p2, f, power = posthoc_power_from_F(F=8.99, df1=1, df2=9, alpha=0.05)
print(f"partial eta^2 = {eta_p2:.3f}, Cohen's f = {f:.3f}, post-hoc power = {power:.3f}")

#%% sample size targets for 80%, 90%, 95% power (using observed f)

df1_interaction = 1
alpha = 0.05

N80, p80 = N_for_target_power(f, df1_interaction, 0.80, alpha)
N90, p90 = N_for_target_power(f, df1_interaction, 0.90, alpha)
N95, p95 = N_for_target_power(f, df1_interaction, 0.95, alpha)

print(f"N for 80% power: {N80} (achieved power = {p80:.3f})")
print(f"N for 90% power: {N90} (achieved power = {p90:.3f})")
print(f"N for 95% power: {N95} (achieved power = {p95:.3f})")
