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

# actual stats packages
import scipy.stats
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import LeaveOneGroupOut
from statsmodels.stats.anova import AnovaRM


#%% Init variables + set paths

# set paths (I used os package since it's platform independent -- easier for someone to reuse)
root_dir = '/Users/lena/Documents/GitHub/henderson2025' # root_dir for the repo
bold_data_path = os.path.join(root_dir, 'bold_data_henderson2025') # path containing downloaded data from Henderson et al.
stats_output_path = os.path.join(root_dir, 'bold_decoding_anova_results') # output path for stats CSVs
sys.path.append(root_dir) # add the repo to the path

num_participants = 10 # number of participants

# uses list comprehension to make a list of sub IDs to iterate over later
sub_ids = [str(i).zfill(2) for i in range(1, num_participants + 1)] # zero pads that sub IDs up to 2 digits (e.g., '1' becomes '01')
sub_ids_list = np.array(sub_ids)  

boundaries = {"linear1": [[1, 4], [2, 3]], "linear2": [[1, 2], [3, 4]], "nonlinear": [[1, 3], [2, 4]]} # dict for which quartiles correspond to which boundary groupings in the different conditions


#%% Helper functions (run this before the cells below!)

# define a function to load the bold data
def load_bold_main_data(sub_id, data_path):
    """
    loads the trial-averaged bold data and labels for each participant (labels are from the behavioral CSVs)
    
    returns
    -------
    main_data_list: list of length num_rois
        each item stored is a 2d float array (num_trials x num_voxels) of bold activity per ROI
        these data are z-scored within the run and averaged over TR window
    main_labels: pandas df
        contains columns ['run_overall','is_main_grid','quadrant','task']
    roi_names: list of length num_rois
        contains ROI names (strings) that align with the order of main_data_list
    """
    
    # read in the labels from the behavioral data
    behavioral_data_path = os.path.join(data_path, f"DataBehavior/S{sub_id}", f"S{sub_id}_maintask_preproc_all.csv") # this is the path to each individual participant's behavioral data file
    main_labels = pd.read_csv(behavioral_data_path, index_col=0)
    
    # get the bold data (we converted this from a mat file to a npz file earlier)
    npz_path = os.path.join(data_path, f"S{sub_id.zfill(2)}_main_data.npz")

    # npz should contain: roi_names (list[str]) and arrays keyed by roi name
    packed = np.load(npz_path, allow_pickle=True)
    roi_names = list(packed["roi_names"])
    main_data_list = [packed[roi] for roi in roi_names]

    # so peeking at the original pipeline, the authors mean center the data (subtract per-trial voxel mean)
    for i in range(len(main_data_list)):
        roi_bold_data = main_data_list[i].astype(float, copy=False)
        main_data_list[i] = roi_bold_data - roi_bold_data.mean(axis=1, keepdims=True)

    return main_data_list, main_labels, roi_names

 
#%% Load in fMRI data














#%% Conduct decoding analysis







#%% run ANOVA (maybe do this in R)
