#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  3 14:43:25 2025

@author: lenakemmelmeier

Reproducing the binary decoding anova result from Henderson et al. (2025)
"""

#%% set up environment - load in appropriate packages, setting paths

import numpy as np
import os # path handling, directory operations
import pandas as pd # trial-wise labels, long-format results
from pathlib import Path # cleaner path handling
import math

from sklearn.model_selection import LeaveOneGroupOut # run-wise cross-validation splits
from sklearn.linear_model import LogisticRegressionCV # logistic regression with C selection
from statsmodels.stats.anova import AnovaRM # repeated-measures anova

from scipy.stats import sem # standard error of the mean
import scipy.stats as sstats # z-scoring and other stats helpers
import scipy.io as spio # loading older-style .mat files (non v7.3)
from scipy.io.matlab import mat_struct # struct type for matlab objects
import h5py # reading v7.3 matlab files (HDF5)
import matplotlib.pyplot as plt

import warnings # suppress noisy but harmless warnings

# silence some version-related warnings from scipy / sklearn that we do not care about.... given this is a reproducibility check
warnings.filterwarnings("ignore", category=DeprecationWarning, module="scipy.io.matlab")
warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn.linear_model._logistic")

# set a global seed so things are reproducible/deterministic
np.random.seed(0)

# set up project-level paths (assumes this script lives in decoding_reproducibility/)
root_dir = Path(__file__).resolve().parents[1]
data_root = os.path.join(root_dir, "data")

stats_output_dir = os.path.join(root_dir, "bold_decoding_anova_results")
fig_out_dir = os.path.join(stats_output_dir, "figs")
os.makedirs(fig_out_dir, exist_ok=True)

# make sure we're in the project root dir!!
os.chdir(root_dir)

#%% init variables (subject list, boundaries, task mapping)

# ids of the ten subjects included in the decoding analysis
num_participants = 10
sub_ids = [str(i).zfill(2) for i in range(1, num_participants + 1)]

# mapping from quadrant labels (1–4) to two-category boundaries for each task
# each boundary has two groups/categories
boundaries = {
    "linear1": [[1, 4], [2, 3]],
    "linear2": [[1, 2], [3, 4]],
    "nonlinear": [[1, 3], [2, 4]],
}

# map from integer task id in the behavioral files to readable task names
task_id_to_name = {1: "linear1", 2: "linear2", 3: "nonlinear", 4: "repeat"}

#%% helper functions (mat conversion, data wrangling, decoding, stats, plotting)

def mat_to_list(ndarray):
    """
    recursively convert matlab cell arrays (ndarrays) into nested python lists
    """

    elem_list = []

    # iterate over each element in the matlab cell array
    for sub_elem in ndarray:

        # if the element is another struct, convert it via mat_to_dict
        if isinstance(sub_elem, mat_struct):
            elem_list.append(mat_to_dict(sub_elem))

        # if the element is another ndarray, recurse (nested cell arrays)
        elif isinstance(sub_elem, np.ndarray):
            elem_list.append(mat_to_list(sub_elem))

        # otherwise keep the raw scalar / string / number
        else:
            elem_list.append(sub_elem)

    return elem_list


def mat_to_dict(matobj):
    """
    recursively convert a matlab struct into a nested python dict
    """

    d = {}

    # matlab structs expose their fields via _fieldnames
    for field_name in matobj._fieldnames:
        elem = matobj.__dict__[field_name]

        # nested struct -> recurse
        if isinstance(elem, mat_struct):
            d[field_name] = mat_to_dict(elem)

        # cell array or array of structs -> handle via mat_to_list
        elif isinstance(elem, np.ndarray):
            d[field_name] = mat_to_list(elem)

        # any other type (numeric, string, etc.) -> copy directly
        else:
            d[field_name] = elem

    return d


def load_samplefile_h5py(sample_fn):
    """
    load a samplefile (mat v7.3) via h5py and collect roi metadata....
    """

    samples = {}

    # these are the dense arrays we care about from the samplefile
    keys_do = ["all_vox_concat", "samplesMain", "samplesRep"]

    # read the large dense arrays once
    with h5py.File(sample_fn, "r") as f:
        for kk in keys_do:
            samples[kk] = np.array(f[kk])

    # roi metadata and flags (multi-dimensional structure in the file)
    rois_keys = ["voxel_inds", "is_md", "is_motor", "is_visual"]
    rois_dict = {}

    # canonical roi and hemisphere labels used in the original paper
    roi_names_source = [
        "V1", "V2", "V3", "V3AB", "hV4",
        "IPS0", "IPS1", "IPS2", "IPS3",
        "LO1", "LO2",
    ]
    hemis = ["lh", "rh"]

    n_rois = len(roi_names_source)
    n_hemis = len(hemis)

    # preallocate nested roi structure (roi x hemisphere)
    for rk in rois_keys:
        rois_dict[rk] = [[[] for _ in range(n_hemis)] for _ in range(n_rois)]

    all_vox_list = []

    # walk through each roi/hemisphere and pull voxel indices plus metadata
    with h5py.File(sample_fn, "r") as f:
        for rk in rois_keys:
            for rr in range(n_rois):
                for hh in range(n_hemis):
                    # get a reference to the roi data, then dereference to an array
                    ref = f["ROIs"][rk][rr][hh]
                    rois_dict[rk][rr][hh] = np.array(f[ref]).astype(int)

                    # for voxel indices, collect into a list so we can sanity check later
                    if rk == "voxel_inds":
                        a = np.array(f[ref]).astype(int)

                        # only store 2d index arrays (skip any empty placeholders)
                        if len(a.shape) == 2:
                            all_vox_list.append(a)

    # sanity check: concatenated voxel list should match all_vox_concat index column
    all_vox_list = np.concatenate(all_vox_list, axis=0)

    # every voxel index should be unique
    assert len(np.unique(all_vox_list)) == len(all_vox_list)

    # and they should match the index column in all_vox_concat
    assert np.all(np.unique(all_vox_list) == samples["all_vox_concat"][:, 0])

    samples["ROIs"] = rois_dict
    samples["ROI_names"] = roi_names_source
    samples["hemis"] = hemis

    return samples


def load_main_data(sub_id, make_time_resolved=False, use_big_ips=True, concat_ips=True):
    """
    load and preprocess main-task bold data for one subject
    """

    ss = int(sub_id)

    # choose whether to use the "bigIPS" samplefile or the original one (here, we will only use bigIPS)
    if use_big_ips:
        sample_fn = os.path.join(data_root, "Samples", f"SampleFile_bigIPS_S{ss:02d}.mat")

    else:
        sample_fn = os.path.join(data_root, "Samples", f"SampleFile_S{ss:02d}.mat")

    samples = load_samplefile_h5py(sample_fn)

    # timing file from geteventtiming.m (matlab struct, converted to dict)
    timing_fn = os.path.join(data_root, "Samples", f"TimingFile_S{ss:02d}.mat")
    timing = spio.loadmat(timing_fn, squeeze_me=True, struct_as_record=False)
    main = mat_to_dict(timing["main"])

    # trial-wise behavioral csv (already preprocessed by original authors)
    behav_fn = os.path.join(
        data_root,
        "DataBehavior",
        f"S{ss:02d}",
        f"S{ss:02d}_maintask_preproc_all.csv",
    )
    bdat = pd.read_csv(behav_fn, index_col=0)

    # tr counts differ slightly for early vs later subjects (scanner protocol change)
    if ss <= 7:
        n_trs = 327 - 16
        avg_trs_targ = [4, 7] # when to average within a run relative to trial onset
        n_trs_concat = 14 # how many trs to keep per trial if making time-resolved data

    else:
        n_trs = 201
        avg_trs_targ = [2, 5]
        n_trs_concat = 9

    n_trials_per_run = 48 # fixed by design of the main task

    roi_names = samples["ROI_names"]
    n_rois = len(roi_names)
    n_hemis = len(samples["hemis"])

    main_data = [] # list of [trials x voxels] arrays, one per roi
    main_data_by_tr = [] # optional [trials x time x voxels] arrays (if requested)

    # loop over rois and hemispheres, then build a single matrix per roi
    for rr in range(n_rois):
        dat_this_roi = []

        for hh in range(n_hemis):
            # voxel indices belonging to this roi and hemisphere
            roi_inds_num = np.array(samples["ROIs"]["voxel_inds"][rr][hh])

            # boolean mask for rows in all_vox_concat that belong to this roi/hemisphere
            inds_this_hemi = np.isin(np.array(samples["all_vox_concat"]), roi_inds_num)[:, 0]

            if np.sum(inds_this_hemi) > 0:
                # samplesMain: [voxels x trs] -> transpose to [trs x voxels]
                dat_this_hemi = samples["samplesMain"][inds_this_hemi, :].T
                dat_this_roi.append(dat_this_hemi)

        # concatenate across hemispheres to get one [trs x voxels] array per roi
        dat_this_roi = np.concatenate(dat_this_roi, axis=1)
        n_vox = dat_this_roi.shape[1]

        # make sure runs and trs match what we expect from the timing file
        n_trs_total = dat_this_roi.shape[0]
        assert np.mod(n_trs_total, n_trs) == 0

        n_runs = int(n_trs_total / n_trs)
        assert n_trs_total == len(main["RunLabels"])
        assert n_runs == len(np.unique(main["RunLabels"]))

        # find trial onset indices from the event labels
        event_diff = np.diff(np.array([0] + list(main["EventLabels"])))
        event_diff_reshaped = np.reshape(event_diff, [n_trs, n_runs], order="F")
        trial_onset_bool = event_diff_reshaped == 1 # event label changed from 0 → 1
        trial_onset_bool = np.reshape(trial_onset_bool, [n_trs * n_runs, 1], order="F")
        trial_onset_num = np.where(trial_onset_bool)[0]

        n_trials = n_runs * n_trials_per_run
        assert len(trial_onset_num) == n_trials

        # sanity checks vs behavioral csv (only need to do this once per subject)
        if rr == 0:
            # response labels: convert 0 to nan so we can match missing values
            resp1 = np.array(main["ResponseLabels"])[trial_onset_num].astype(float)
            resp1[resp1 == 0] = np.nan

            resp2 = np.array(bdat["resp"])
            has_nans = np.isnan(resp1)

            # missing responses should agree between matlab timing struct and csv
            assert np.all(np.isnan(resp2[has_nans]))
            assert np.all(resp1[~has_nans] == resp2[~has_nans])

            # grid vs non-grid trials
            vals1 = np.array(main["IsMainLabels"])[trial_onset_num]
            vals2 = np.array(bdat["is_main_grid"])
            assert np.all(vals1 == vals2)

            # point coordinates (x and y)
            vals1 = np.array(main["PointLabels"])[:, 0][trial_onset_num]
            vals2 = np.array(bdat["ptx"])
            assert np.all(vals1 == vals2)

            vals1 = np.array(main["PointLabels"])[:, 1][trial_onset_num]
            vals2 = np.array(bdat["pty"])
            assert np.all(vals1 == vals2)

            # boundary, map, session, run, trial indices
            vals1 = np.array(main["BoundLabels"])[trial_onset_num]
            vals2 = np.array(bdat["task"])
            assert np.all(vals1 == vals2)

            vals1 = np.array(main["MapLabels"])[trial_onset_num]
            vals2 = np.array(bdat["map"])
            assert np.all(vals1 == vals2)

            vals1 = np.array(main["SessLabels"])[trial_onset_num]
            vals2 = np.array(bdat["sess"])
            assert np.all(vals1 == vals2)

            vals1 = np.array(main["RunLabels"])[trial_onset_num]
            vals2 = np.array(bdat["run_overall"])
            assert np.all(vals1 == vals2)

            vals1 = np.array(main["TrialLabels"])[trial_onset_num]
            vals2 = np.array(bdat["trial_overall"])
            assert np.all(vals1 == vals2)

        # z-score bold per run (match original preprocessing choices)
        for run in np.unique(main["RunLabels"]):
            run_inds = main["RunLabels"] == run
            assert np.sum(run_inds) == n_trs

            dat_this_roi[run_inds, :] = sstats.zscore(dat_this_roi[run_inds, :], axis=0)

        # allocate arrays for trial-averaged and time-resolved data
        dat_avg_targ = np.full((n_trials, n_vox), np.nan)

        if make_time_resolved:
            dat_by_tr = np.full((n_trials, n_trs_concat, n_vox), np.nan)

        else:
            dat_by_tr = None

        trial_counter = -1

        # build trial-wise averaged windows (and, if enabled, time-resolved data)
        for run in np.unique(main["RunLabels"]):
            run_inds = main["RunLabels"] == run
            assert np.sum(run_inds) == n_trs

            cur_dat = dat_this_roi[run_inds, :] # bold for this run only

            # trial onsets within this run
            these_targ_onsets = np.where(trial_onset_bool[run_inds])[0]
            assert len(these_targ_onsets) == n_trials_per_run

            for tt in these_targ_onsets:
                window_start = tt + avg_trs_targ[0]
                window_stop = tt + avg_trs_targ[1]

                # only keep windows that fully fit within the run
                if window_stop < n_trs:
                    trial_counter += 1

                    # average bold over the chosen tr window for this trial
                    dat_avg_targ[trial_counter, :] = np.mean(
                        cur_dat[window_start:window_stop, :],
                        axis=0,
                    )

                    # optional: also keep the full tr-by-tr timecourse per trial
                    if make_time_resolved:
                        for tr in range(n_trs_concat):
                            dat_by_tr[trial_counter, tr, :] = cur_dat[tt + tr, :]

        # verify that all trials were filled and no nans remain
        assert trial_counter == n_trials - 1
        assert not np.any(np.isnan(dat_avg_targ.ravel()))

        if make_time_resolved:
            assert not np.any(np.isnan(dat_by_tr.ravel()))

        main_data.append(dat_avg_targ)
        main_data_by_tr.append(dat_by_tr)

    # if concat_ips is toggled on, combine IPS0–IPS3 into one "IPSall" roi
    if concat_ips:
        ips_roi = [5, 6, 7, 8]

        # keep early visual + LO rois, add a combined IPS row at the end
        roi_names = roi_names[0:5] + roi_names[9:] + ["IPSall"]

        # concatenate IPS voxel matrices horizontally
        main_data_new = main_data[0:5] + main_data[9:]
        ips_concat = np.concatenate([main_data[rr] for rr in ips_roi], axis=1)
        main_data_new.append(ips_concat)
        main_data = main_data_new

        # if time-resolved data were requested, also combine IPS timecourses
        if make_time_resolved:
            main_data_by_tr_new = main_data_by_tr[0:5] + main_data_by_tr[9:]
            ips_concat = np.concatenate([main_data_by_tr[rr] for rr in ips_roi], axis=2)
            main_data_by_tr_new.append(ips_concat)
            main_data_by_tr = main_data_by_tr_new

    main_labels = bdat

    # only return 3 items to match how callers are written
    return main_data, main_labels, roi_names


def load_repeat_data(sub_id, make_time_resolved=False, use_big_ips=True, concat_ips=True):
    """
    load and preprocess repeat-task bold data for one subject
    """

    ss = int(sub_id)

    # choose which samplefile to use, same switch as in load_main_data
    if use_big_ips:
        sample_fn = os.path.join(data_root, "Samples", f"SampleFile_bigIPS_S{ss:02d}.mat")

    else:
        sample_fn = os.path.join(data_root, "Samples", f"SampleFile_S{ss:02d}.mat")

    samples = load_samplefile_h5py(sample_fn)

    # timing and behavior for the repeat task
    timing_fn = os.path.join(data_root, "Samples", f"TimingFile_S{ss:02d}.mat")
    timing = spio.loadmat(timing_fn, squeeze_me=True, struct_as_record=False)
    rep = mat_to_dict(timing["rep"])

    behav_fn = os.path.join(
        data_root,
        "DataBehavior",
        f"S{ss:02d}",
        f"S{ss:02d}_reptask_preproc_all.csv",
    )
    bdat = pd.read_csv(behav_fn, index_col=0)

    # again, early vs later subjects differ slightly in number of trs per run
    if ss <= 7:
        n_trs = 329 - 16
        avg_trs_targ = [4, 7]
        n_trs_concat = 14

    else:
        n_trs = 203
        avg_trs_targ = [2, 5]
        n_trs_concat = 9

    n_trials_per_run = 48

    roi_names = samples["ROI_names"]
    n_rois = len(roi_names)
    n_hemis = len(samples["hemis"])

    rep_data = [] # [trials x voxels] per roi
    rep_data_by_tr = [] # [trials x time x voxels] per roi if requested

    # loop over rois and hemispheres, then build a single matrix per roi
    for rr in range(n_rois):
        dat_this_roi = []

        for hh in range(n_hemis):
            roi_inds_num = np.array(samples["ROIs"]["voxel_inds"][rr][hh])
            inds_this_hemi = np.isin(np.array(samples["all_vox_concat"]), roi_inds_num)[:, 0]

            if np.sum(inds_this_hemi) > 0:
                dat_this_hemi = samples["samplesRep"][inds_this_hemi, :].T
                dat_this_roi.append(dat_this_hemi)

        dat_this_roi = np.concatenate(dat_this_roi, axis=1)
        n_vox = dat_this_roi.shape[1]

        n_trs_total = dat_this_roi.shape[0]
        assert np.mod(n_trs_total, n_trs) == 0

        n_runs = int(n_trs_total / n_trs)
        assert n_trs_total == len(rep["RunLabels"])
        assert n_runs == len(np.unique(rep["RunLabels"]))

        # compute trial onset indices for the repeat task
        event_diff = np.diff(np.array([0] + list(rep["EventLabels"])))
        event_diff_reshaped = np.reshape(event_diff, [n_trs, n_runs], order="F")
        trial_onset_bool = event_diff_reshaped == 1
        trial_onset_bool = np.reshape(trial_onset_bool, [n_trs * n_runs, 1], order="F")
        trial_onset_num = np.where(trial_onset_bool)[0]

        n_trials = n_runs * n_trials_per_run
        assert len(trial_onset_num) == n_trials

        # quick alignment checks vs behavioral csv (as above)
        if rr == 0:
            resp1 = np.array(rep["ResponseLabels"])[trial_onset_num].astype(float)
            resp1[resp1 == 0] = np.nan

            resp2 = np.array(bdat["resp"])
            has_nans = np.isnan(resp1)

            assert np.all(np.isnan(resp2[has_nans]))
            assert np.all(resp1[~has_nans] == resp2[~has_nans])

            vals1 = np.array(rep["IsMainLabels"])[trial_onset_num]
            vals2 = np.array(bdat["is_main_grid"])
            assert np.all(vals1 == vals2)

            vals1 = np.array(rep["PointLabels"])[:, 0][trial_onset_num]
            vals2 = np.array(bdat["ptx"])
            assert np.all(vals1 == vals2)

            vals1 = np.array(rep["PointLabels"])[:, 1][trial_onset_num]
            vals2 = np.array(bdat["pty"])
            assert np.all(vals1 == vals2)

            vals1 = np.array(rep["MapLabels"])[trial_onset_num]
            vals2 = np.array(bdat["map"])
            assert np.all(vals1 == vals2)

            vals1 = np.array(rep["SessLabels"])[trial_onset_num]
            vals2 = np.array(bdat["sess"])
            assert np.all(vals1 == vals2)

            vals1 = np.array(rep["RunLabels"])[trial_onset_num]
            vals2 = np.array(bdat["run_overall"])
            assert np.all(vals1 == vals2)

            vals1 = np.array(rep["TrialLabels"])[trial_onset_num]
            vals2 = np.array(bdat["trial_overall"])
            assert np.all(vals1 == vals2)

        # z-score bold per run (repeat task)
        for run in np.unique(rep["RunLabels"]):
            run_inds = rep["RunLabels"] == run
            assert np.sum(run_inds) == n_trs

            dat_this_roi[run_inds, :] = sstats.zscore(dat_this_roi[run_inds, :], axis=0)

        dat_avg_targ = np.full((n_trials, n_vox), np.nan)

        if make_time_resolved:
            dat_by_tr = np.full((n_trials, n_trs_concat, n_vox), np.nan)

        else:
            dat_by_tr = None

        trial_counter = -1

        # build trial-wise averaged windows (and optional time-resolved data)
        for run in np.unique(rep["RunLabels"]):
            run_inds = rep["RunLabels"] == run
            assert np.sum(run_inds) == n_trs

            cur_dat = dat_this_roi[run_inds, :]

            these_targ_onsets = np.where(trial_onset_bool[run_inds])[0]
            assert len(these_targ_onsets) == n_trials_per_run

            for tt in these_targ_onsets:
                window_start = tt + avg_trs_targ[0]
                window_stop = tt + avg_trs_targ[1]

                if window_stop < n_trs:
                    trial_counter += 1

                    dat_avg_targ[trial_counter, :] = np.mean(
                        cur_dat[window_start:window_stop, :],
                        axis=0,
                    )

                    if make_time_resolved:
                        for tr in range(n_trs_concat):
                            dat_by_tr[trial_counter, tr, :] = cur_dat[tt + tr, :]

        assert trial_counter == n_trials - 1
        assert not np.any(np.isnan(dat_avg_targ.ravel()))

        if make_time_resolved:
            assert not np.any(np.isnan(dat_by_tr.ravel()))

        rep_data.append(dat_avg_targ)
        rep_data_by_tr.append(dat_by_tr)

    # if concat_ips is toggled on, combine IPS0–IPS3 into one "IPSall" roi
    if concat_ips:
        ips_roi = [5, 6, 7, 8]
        roi_names = roi_names[0:5] + roi_names[9:] + ["IPSall"]

        rep_data_new = rep_data[0:5] + rep_data[9:]
        ips_concat = np.concatenate([rep_data[rr] for rr in ips_roi], axis=1)
        rep_data_new.append(ips_concat)
        rep_data = rep_data_new

        if make_time_resolved:
            rep_data_by_tr_new = rep_data_by_tr[0:5] + rep_data_by_tr[9:]
            ips_concat = np.concatenate([rep_data_by_tr[rr] for rr in ips_roi], axis=2)
            rep_data_by_tr_new.append(ips_concat)
            rep_data_by_tr = rep_data_by_tr_new

    rep_labels = bdat

    return rep_data, rep_labels, roi_names


def rename_ips(roi_names):
    """
    rename concatenated ips roi from 'ipsall' to 'ips' - this is just for plotting!!
    """

    # this makes plots and tables look cleaner
    return ["IPS" if r == "IPSall" else r for r in roi_names]


def make_binary_labels(q_arr, boundary_name):
    """
    map quadrant labels onto binary category labels for a given boundary
    """

    # g0 and g1 each contain a list of quadrants belonging to that category
    g0, g1 = boundaries[boundary_name]

    # start with an invalid label (-1) so we can check for unassigned trials
    y = np.full(q_arr.shape, -1, dtype=int)

    # map quadrants in g0 to class 1 and g1 to class 2
    y[np.isin(q_arr, g0)] = 1
    y[np.isin(q_arr, g1)] = 2

    return y


def make_near_far_masks(dist1, dist2, dist3, boundary_name):
    """
    build near and far boolean masks based on the appropriate distance column
    """

    # near/far is defined relative to the distance for the boundary in question
    if boundary_name == "linear1":
        near = np.isclose(dist1, 0.8)
        far = np.isclose(dist1, 2.4)

        return near, far

    if boundary_name == "linear2":
        near = np.isclose(dist2, 0.8)
        far = np.isclose(dist2, 2.4)

        return near, far

    if boundary_name == "nonlinear":
        near = np.isclose(dist3, 0.8)
        far = np.isclose(dist3, 2.4)

        return near, far

    # default: no trials marked near or far (should not be used in practice)
    near = np.zeros_like(dist1, dtype=bool)
    far = np.zeros_like(dist1, dtype=bool)

    return near, far


def decode_within_task_one_roi_replica(
    data_concat,
    quad_labs,
    task_labs,
    cv_labs,
    is_main_grid,
    task_id,
    boundary_name,
):
    """
    run within-task binary decoding for one roi using all main-grid trials
    """

    # pick out only trials belonging to the task of interest
    tinds = task_labs == task_id

    if not np.any(tinds):
        return np.nan

    x_data = data_concat[tinds, :]
    q = quad_labs[tinds]
    runs = cv_labs[tinds]
    mg = is_main_grid[tinds]

    # map quadrants to binary category labels for this boundary
    y = make_binary_labels(q, boundary_name)

    # if any trial remained unassigned, bail out
    if (y <= 0).any():
        return np.nan

    # mean-center each trial across voxels (match original implementation of authors...)
    x_data = x_data - x_data.mean(axis=1, keepdims=True)
    x_data = np.asarray(x_data, dtype=np.float64)

    n_trials = x_data.shape[0]
    pred = np.full(n_trials, np.nan)

    # same log-spaced C grid as in the original code
    c_values = np.logspace(-9, 1, 20)
    logo = LeaveOneGroupOut()

    # outer cv is run-wise; inner cv reused for C selection
    for cv in np.unique(runs):
        tr = (runs != cv) & mg # training trials in other runs, main grid only
        te = runs == cv # test trials in this run

        # ensure enough data and at least two classes in the training set
        if tr.sum() < 4 or np.unique(y[tr]).size < 2 or te.sum() == 0:
            continue

        inner_groups = runs[tr]
        inner_cv = logo.split(x_data[tr], y[tr], groups=inner_groups)

        clf = LogisticRegressionCV(
            cv=inner_cv,
            Cs=c_values,
            multi_class="multinomial",
            solver="lbfgs",
            penalty="l2",
            n_jobs=-1,
            max_iter=20000,
            tol=1e-3,
        )
        clf.fit(x_data[tr], y[tr])

        # store predictions for trials in this left-out run
        pred[te] = clf.predict(x_data[te])

    # if some runs were skipped, only score valid main-grid trials
    if np.isnan(pred).any():
        valid = (~np.isnan(pred)) & mg

        if valid.sum() == 0:
            return np.nan

        return (pred[valid] == y[valid]).mean()

    return (pred[mg] == y[mg]).mean()


def decode_with_subset(
    data_concat,
    quad_labs,
    task_labs,
    cv_labs,
    is_main_grid,
    task_id,
    boundary_name,
    subset_mask,
):
    """
    run within-task binary decoding for one roi on a subset (e.g., near OR far trials)
    """

    # task-specific mask
    tinds = task_labs == task_id

    if not np.any(tinds):
        return np.nan

    x_data = data_concat[tinds, :]
    q = quad_labs[tinds]
    runs = cv_labs[tinds]
    mg = is_main_grid[tinds]

    y = make_binary_labels(q, boundary_name)

    # again, bail out if any trial was not mapped to class 1 or 2
    if (y <= 0).any():
        return np.nan

    # mean-center each trial across voxels
    x_data = x_data - x_data.mean(axis=1, keepdims=True)
    x_data = np.asarray(x_data, dtype=np.float64)

    pred = np.full(len(x_data), np.nan)
    c_values = np.logspace(-9, 1, 20)
    logo = LeaveOneGroupOut()

    # outer cv folds are runs; inner folds reused for C selection
    for cv in np.unique(runs):
        tr = (runs != cv) & mg # training set: main-grid trials in other runs
        te = runs == cv # test set: trials in this run (near or far will be selected later)

        if tr.sum() < 4 or np.unique(y[tr]).size < 2 or te.sum() == 0:
            continue

        inner_groups = runs[tr]
        inner_cv = logo.split(x_data[tr], y[tr], groups=inner_groups)

        clf = LogisticRegressionCV(
            cv=inner_cv,
            Cs=c_values,
            multi_class="multinomial",
            solver="lbfgs",
            penalty="l2",
            n_jobs=-1,
            max_iter=20000,
            tol=1e-3,
        )
        clf.fit(x_data[tr], y[tr])

        pred[te] = clf.predict(x_data[te])

    # restrict accuracy to the subset mask (near or far) and main-grid trials
    subset = subset_mask[tinds] & mg
    valid = (~np.isnan(pred)) & subset

    if valid.sum() == 0:
        return np.nan

    return (pred[valid] == y[valid]).mean()


def run_task_boundary_roi_anova(df_long, out_csv, print_table=False):
    """
    run a 3-way repeated-measures anova over roi × task × boundary (near only)
    """

    df_ = df_long.copy()

    # restrict to linear tasks/boundaries and near trials (as in the original result)
    df_ = df_[
        df_["Task"].isin(["linear1", "linear2"])
        & df_["Boundary"].isin(["linear1", "linear2"])
        & (df_["Dist"] == "near")
    ].copy()

    # drop any rows with missing accuracy values
    df_ = df_.dropna(subset=["ACC"])

    # make sure the within-subjects factors are treated as strings
    for col in ["sub", "ROI", "Task", "Boundary"]:
        df_[col] = df_[col].astype(str)

    aov = AnovaRM(
        df_,
        depvar="ACC",
        subject="sub",
        within=["ROI", "Task", "Boundary"],
    ).fit()

    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    aov.anova_table.to_csv(out_csv)

    if print_table:
        print(aov.anova_table)

    return aov


def effect_sizes_from_f(f_value, df1, df2):
    """
    compute partial eta-squared and cohen's f from an f statistic
    """

    # standard formulas for partial eta^2 and cohen's f
    eta_p2 = (f_value * df1) / (f_value * df1 + df2)
    cohen_f = math.sqrt(eta_p2 / (1.0 - eta_p2))

    return eta_p2, cohen_f


def save_apa_anova_table(anova_result, out_png):
    """
    save an apa-style anova table as a png for the report
    """

    df = anova_result.copy()

    # depending on how the csv is read, the effect column may live in 'unnamed: 0'
    if "Unnamed: 0" in df.columns:
        df = df.rename(columns={"Unnamed: 0": "Effect"})

    else:
        df = df.reset_index().rename(columns={"index": "Effect"})

    def fmt_p(p):
        # format p-values as strings: '< .001' or '0.xxx'
        if pd.isna(p):
            return ""

        if p < 0.001:
            return "< .001"

        return f"{p:.3f}".lstrip("0")

    # basic formatting for df, F, p, and eta
    df["Num DF"] = df["Num DF"].astype(int)
    df["Den DF"] = df["Den DF"].astype(int)
    df["F Value"] = df["F Value"].map(lambda x: f"{x:.2f}")
    df["Pr > F"] = df["Pr > F"].map(fmt_p)
    df["eta_p2"] = df["eta_p2"].map(lambda x: f"{x:.02f}")

    df = df.rename(
        columns={
            "Num DF": "df1",
            "Den DF": "df2",
            "F Value": "F",
            "Pr > F": "p",
            "eta_p2": "np²",
        }
    )
    df = df[["Effect", "df1", "df2", "F", "p", "np²"]]

    # build a simple table figure without grid lines inside
    n_rows = len(df)
    fig, ax = plt.subplots(figsize=(6, 0.6 + 0.3 * n_rows))
    ax.axis("off")

    table = ax.table(
        cellText=df.values,
        colLabels=df.columns,
        loc="center",
        cellLoc="right",
    )

    table.auto_set_font_size(False)
    table.set_fontsize(10)

    for col in range(len(df.columns)):
        table.auto_set_column_width(col)

    n_cols = len(df.columns)

    for (r, c), cell in table.get_celld().items():
        cell.visible_edges = ""
        cell.set_height(0.12)

        if r == 0:
            cell.set_text_props(weight="bold")
            cell.set_facecolor("#F2F2F2")

        else:
            cell.set_facecolor("white")

        if c == 0:
            cell._loc = "left"
            cell._text.set_ha("left")

        else:
            cell._loc = "right"
            cell._text.set_ha("right")

    # draw horizontal lines above header, below header, and at the bottom
    left = table[(0, 0)].get_x()
    right = table[(0, n_cols - 1)].get_x() + table[(0, n_cols - 1)].get_width()

    top_header = table[(0, 0)].get_y() + table[(0, 0)].get_height()
    bottom_header = table[(0, 0)].get_y()
    bottom_table = table[(n_rows, 0)].get_y()

    ax.hlines(top_header, left, right, linewidth=1.5, color="black")
    ax.hlines(bottom_header, left, right, linewidth=1.0, color="black")
    ax.hlines(bottom_table, left, right, linewidth=1.5, color="black")

    fig.tight_layout()
    fig.savefig(out_png, dpi=600, bbox_inches="tight")
    plt.close(fig)


def plot_panel(
    acc_long_fig2,
    roi_order,
    task_order,
    task_labels,
    task_colors,
    chance_y,
    offsets,
    boundary,
    title,
):
    """
    plot one fig. 2-style panel for a given boundary (linear1 / linear2 / nonlinear)
    """

    # filter to the boundary of interest, keeping all tasks in task_order
    df_b = acc_long_fig2.query("Boundary == @boundary and Task in @task_order").copy()

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.axhline(chance_y, color="gray", linewidth=1.0)

    x_base = np.arange(len(roi_order))

    for task in task_order:
        df_t = df_b[df_b["Task"] == task]

        # scatter subject-level accuracies per roi (in gray)
        for _, df_s in df_t.groupby("sub"):
            df_s_roi = df_s.groupby("ROI", as_index=True)["ACC"].mean()
            y = df_s_roi.reindex(roi_order).values
            x = x_base + offsets[task]

            ax.scatter(x, y, s=12, color="#BFBFBF", zorder=1)

        # plot roi-wise mean ± sem in color for each task
        g = df_t.groupby("ROI")["ACC"]
        means = g.mean().reindex(roi_order).values
        errors = g.apply(lambda x: sem(x, nan_policy="omit")).reindex(roi_order).values
        x = x_base + offsets[task]

        ax.errorbar(
            x,
            means,
            yerr=errors,
            fmt="o",
            markersize=8,
            capsize=3,
            linewidth=1.2,
            color=task_colors[task],
            ecolor=task_colors[task],
            label=task_labels[task],
            zorder=3,
        )

    ax.set_xticks(x_base)
    ax.set_xticklabels(roi_order, fontsize=18)
    ax.set_ylim(0.4, 1.0)
    ax.tick_params(axis="y", labelsize=18)
    ax.set_ylabel("Classifier accuracy", fontsize=16)
    ax.set_title(title, fontsize=18)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.legend(frameon=False, loc="upper right")
    fig.tight_layout()

    return fig, ax


def subject_decoding_df_replica(sub_id):
    """
    run within-task binary decoding (near vs far) for all rois and tasks for one subject
    """

    print(f"[decode] started decoding for subject {sub_id}")

    # load main and repeat task data + labels for this subject
    main_rois, main_lab, main_roi_names = load_main_data(sub_id)
    rep_rois, rep_lab, rep_roi_names = load_repeat_data(sub_id)

    # rename "IPSall" to "IPS" for downstream plots
    roi_names = rename_ips(main_roi_names)
    n_rois = len(roi_names)

    # stack main and repeat labels together so that runs can be used as one cv axis
    concat_labels = pd.concat([main_lab, rep_lab], axis=0)

    # run labels for cross-validation (offset repeat runs so they are unique)
    cv_main = main_lab["run_overall"].to_numpy().astype(int)
    cv_rep = rep_lab["run_overall"].to_numpy().astype(int) + cv_main.max()
    cv_labs = np.concatenate([cv_main, cv_rep], axis=0)

    # masks and labels derived from the combined dataframe
    is_main_grid = concat_labels["is_main_grid"].to_numpy().astype(int) == 1
    quad_labs = concat_labels["quadrant"].to_numpy().astype(int)
    task_labs = concat_labels["task"].to_numpy().astype(int)

    # anything that is not 1/2/3 becomes task id 4 ("repeat"/other)
    task_labs[~np.isin(task_labs, [1, 2, 3])] = 4

    # trial-wise distances from each boundary
    dist1 = concat_labels["dist_from_bound1"].to_numpy()
    dist2 = concat_labels["dist_from_bound2"].to_numpy()
    dist3 = concat_labels["dist_from_bound3"].to_numpy()

    # concatenate main and repeat bold data for each roi
    roi_arrays = []

    for ri in range(n_rois):
        x_main = main_rois[ri]
        x_rep = rep_rois[ri]

        roi_arrays.append(np.concatenate([x_main, x_rep], axis=0))

    rows = []

    # loop over roi × task × boundary, then decode near vs far subsets
    for roi_name, x_data in zip(roi_names, roi_arrays):
        for task_id in (1, 2, 3):
            for bname in ("linear1", "linear2", "nonlinear"):
                # near and far are defined differently for each boundary
                near_mask, far_mask = make_near_far_masks(dist1, dist2, dist3, bname)

                acc_near = decode_with_subset(
                    x_data,
                    quad_labs,
                    task_labs,
                    cv_labs,
                    is_main_grid,
                    task_id,
                    bname,
                    subset_mask=near_mask,
                )

                acc_far = decode_with_subset(
                    x_data,
                    quad_labs,
                    task_labs,
                    cv_labs,
                    is_main_grid,
                    task_id,
                    bname,
                    subset_mask=far_mask,
                )

                rows.append(
                    [sub_id, roi_name, task_id_to_name[task_id], bname, "near", acc_near]
                )
                rows.append(
                    [sub_id, roi_name, task_id_to_name[task_id], bname, "far", acc_far]
                )

    print(f"[decode] finished decoding for subject {sub_id}")

    return pd.DataFrame(rows, columns=["sub", "ROI", "Task", "Boundary", "Dist", "ACC"])


#%% one-time near/far sanity check (subject 01)

# this sanity check confirms that near/far counts match our expectations 
test_sub = sub_ids[0]

main_rois_chk, main_lab_chk, _ = load_main_data(test_sub)
rep_rois_chk, rep_lab_chk, _ = load_repeat_data(test_sub)

labels_chk = pd.concat([main_lab_chk, rep_lab_chk], axis=0)

is_main_grid_chk = labels_chk["is_main_grid"].to_numpy().astype(int) == 1
d1_chk = labels_chk["dist_from_bound1"].to_numpy()
d2_chk = labels_chk["dist_from_bound2"].to_numpy()
d3_chk = labels_chk["dist_from_bound3"].to_numpy()

for bname, (dist1, dist2, dist3) in zip(
    ["linear1", "linear2", "nonlinear"],
    [(d1_chk, d2_chk, d3_chk), (d1_chk, d2_chk, d3_chk), (d1_chk, d2_chk, d3_chk)],
):
    near_mask_chk, far_mask_chk = make_near_far_masks(dist1, dist2, dist3, bname)

    near_n = (near_mask_chk & is_main_grid_chk).sum()
    far_n = (far_mask_chk & is_main_grid_chk).sum()

    print(f"[near/far sanity] sub {test_sub}, {bname}: near={near_n}, far={far_n}")

# if the data are wrangled correctly, there should be....
# 768 near and 768 far trials for both linear1 and linear2
# 1152 near trials and 384 far trials for the nonlinear task

#%% decode data → near/far results + fig. 2 copy

# run decoding for each subject and stack results into one long dataframe
acc_rows_fig2 = [subject_decoding_df_replica(s) for s in sub_ids]
acc_long_fig2 = pd.concat(acc_rows_fig2, ignore_index=True)

# save out near/far data and a separate copy that mirrors fig. 2 organization
acc_long_fig2.to_csv(
    os.path.join(stats_output_dir, "binary_withintask_ACC_long_NEARFAR_replica.csv"),
    index=False,
)

acc_long_fig2.to_csv(
    os.path.join(stats_output_dir, "binary_withintask_ACC_long_FIG2_replica.csv"),
    index=False,
)


#%% run 3-way anova (near only, linear tasks & boundaries only)

# read the near/far csv back in (handy if we want to re-run stats only)
acc_long_nearfar = pd.read_csv(
    os.path.join(stats_output_dir, "binary_withintask_ACC_long_NEARFAR_replica.csv")
)

# fit roi × task × boundary anova and save the table
aov = run_task_boundary_roi_anova(
    acc_long_nearfar,
    out_csv=os.path.join(stats_output_dir, "anova_table_LINEAR_NEARONLY_replica.csv"),
)

anova_result = pd.read_csv(
    os.path.join(stats_output_dir, "anova_table_LINEAR_NEARONLY_replica.csv")
)

# add partial eta-squared and cohen's f for each effect in the table
eta_vals = []
f_vals = []

for f_value, d1, d2 in zip(
    anova_result["F Value"],
    anova_result["Num DF"],
    anova_result["Den DF"],
):
    eta_p2, cohen_f = effect_sizes_from_f(f_value, d1, d2)

    eta_vals.append(eta_p2)
    f_vals.append(cohen_f)

anova_result["eta_p2"] = eta_vals
anova_result["cohens_f"] = f_vals

anova_result.to_csv(
    os.path.join(
        stats_output_dir,
        "anova_table_LINEAR_NEARONLY_replica_with_effectsizes.csv",
    ),
    index=False,
)

print(anova_result)


#%% apa-style anova table (image; no cohen's f column)

# create a publication-style table figure for the anova results
save_apa_anova_table(
    anova_result,
    out_png=os.path.join(fig_out_dir, "Table1_LINEAR_NEARONLY_replica_APA.png"),
)


#%% making our version of the authors' figure 2a–c plots

# re-load the fig. 2-style long dataframe (if needed)
acc_long_fig2 = pd.read_csv(
    os.path.join(stats_output_dir, "binary_withintask_ACC_long_FIG2_replica.csv")
)

# order of rois and tasks to match the paper
roi_order = ["V1", "V2", "V3", "V3AB", "hV4", "LO1", "LO2", "IPS"]
task_order = ["linear1", "linear2", "nonlinear"]

task_labels = {
    "linear1": "Linear-1 Task",
    "linear2": "Linear-2 Task",
    "nonlinear": "Nonlinear Task",
}

task_colors = {
    "linear1": "#1f4e79",
    "linear2": "#2e86c1",
    "nonlinear": "#76d7c4",
}

chance_y = 0.5 # chance accuracy for a binary classifier

# horizontal offsets so condition means do not overlap
offsets = {
    "linear1": -0.18,
    "linear2": 0.0,
    "nonlinear": +0.18,
}

# one panel per boundary, matching the original fig. 2a–c layout
panels = [
    ("linear1", 'A  Binary classifier: Predict "Linear-1" category'),
    ("linear2", 'B  Binary classifier: Predict "Linear-2" category'),
    ("nonlinear", 'C  Binary classifier: Predict "Nonlinear" category'),
]

# loop over panels and save each one as a separate png
for boundary, title in panels:
    fig, ax = plot_panel(
        acc_long_fig2=acc_long_fig2,
        roi_order=roi_order,
        task_order=task_order,
        task_labels=task_labels,
        task_colors=task_colors,
        chance_y=chance_y,
        offsets=offsets,
        boundary=boundary,
        title=title,
    )

    out_png = os.path.join(fig_out_dir, f"Figure2_{boundary}_replica_DOTSEM.png")
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.show()