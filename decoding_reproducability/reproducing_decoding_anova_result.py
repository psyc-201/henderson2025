#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  3 14:43:25 2025

@author: lenakemmelmeier
"""

#%% Set up environment - load in appropiate packages

import numpy as np
import os, sys # used for configuring paths
import pandas as pd
import sklearn
import sklearn.svm, sklearn.discriminant_analysis, sklearn.linear_model
import time
from joblib import effective_n_jobs
import datetime
import scipy.stats


# notes to self -- do not need dprime; just getting decoding acc, not d-prime
# don't need any special helper functions


#%% Init variables + set paths

# note to self: change this below so it can read where the repo got cloned into
# also... should probably not be loading the data onto git
# I should add into the ReadMe where people can go download this instead of posting their data
bold_data_path = '/Users/lena/Documents/GitHub/henderson2025/bold_data_henderson2025/' # add the path with the original data to path
sys.path.append(bold_data_path)

num_participants = 10 # number of participants
sub_ids = [str(i) for i in range(1, num_participants + 1)] # uses list comprehension to make a list of sub_ids to iterate over later
sub_ids_list = np.array(sub_ids)

#%% Helper functions (run this before the cells below!)

# define a function to load the main task bold data in
def load_bold_main_data(sub_id, data_path, ):
    
    # load the 

#%% Load in fMRI data






#%% Conduct decoding analysis







#%% run ANOVA (maybe do this in R)
