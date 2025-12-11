#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  3 14:43:25 2025

@author: lenakemmelmeier
"""

#%% Set up environment - load in appropriate packages

import numpy as np
import os
import pandas as pd
from pathlib import Path
import math

from sklearn.model_selection import LeaveOneGroupOut
from sklearn.linear_model import LogisticRegressionCV
from statsmodels.stats.anova import AnovaRM

from scipy.stats import sem
import scipy.stats as sstats
import scipy.io as spio
from scipy.io.matlab import mat_struct
import h5py
import matplotlib.pyplot as plt

import warnings

# for reproducibility sake, I mirrored how the original authors' loaded in data + went about binary classification
# so I will silence these warnings
warnings.filterwarnings(
    "ignore",
    category=DeprecationWarning,
    module="scipy.io.matlab"
)
warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    module="sklearn.linear_model._logistic"
)

np.random.seed(0) # setting a seed so it's deterministic

root_dir = Path(__file__).resolve().parents[1]
DATA_ROOT = os.path.join(root_dir, "data")

stats_output_path = os.path.join(root_dir, 'bold_decoding_anova_results')
fig_outdir = os.path.join(stats_output_path, "figs")
os.makedirs(fig_outdir, exist_ok=True)

# ensure we are in project root (for relative paths to work the same way)
os.chdir(root_dir)

#%% definitions of helper functions

def _tolist(ndarray):
    """
    Recursively convert MATLAB cell arrays (loaded as ndarrays) into Python lists.
    """
    elem_list = []
    for sub_elem in ndarray:
        if isinstance(sub_elem, mat_struct):  # <-- changed here
            elem_list.append(_todict(sub_elem))
        elif isinstance(sub_elem, np.ndarray):
            elem_list.append(_tolist(sub_elem))
        else:
            elem_list.append(sub_elem)
    return elem_list

def _todict(matobj):
    """
    Recursively convert MATLAB struct objects into nested Python dicts.
    """
    d = {}
    for strg in matobj._fieldnames:
        elem = matobj.__dict__[strg]
        if isinstance(elem, mat_struct):  # <-- and here
            d[strg] = _todict(elem)
        elif isinstance(elem, np.ndarray):
            d[strg] = _tolist(elem)
        else:
            d[strg] = elem
    return d

def load_samplefile_h5py(sample_fn):
    """
    Load a SampleFile (MAT v7.3) into Python using h5py.
    Returns a dict with:
        - 'all_vox_concat'
        - 'samplesMain'
        - 'samplesRep'
        - 'ROIs' (nested dict)
        - 'ROI_names'
        - 'hemis'
    """
    samples = {}
    keys_do = ['all_vox_concat', 'samplesMain', 'samplesRep']

    with h5py.File(sample_fn, 'r') as f:
        for kk in keys_do:
            samples[kk] = np.array(f[kk])

    rois_keys = ['voxel_inds', 'is_md', 'is_motor', 'is_visual']
    rois_dict = {}
    ROI_names = ['V1','V2','V3','V3AB','hV4','IPS0','IPS1','IPS2','IPS3','LO1','LO2']
    hemis = ['lh', 'rh']
    n_rois = 11
    n_hemis = 2

    for rk in rois_keys:
        rois_dict[rk] = [[[] for _ in range(n_hemis)] for _ in range(n_rois)]

    all_vox_list = []

    with h5py.File(sample_fn, 'r') as f:
        for rk in rois_keys:
            for rr in range(n_rois):
                for hh in range(n_hemis):
                    ref = f['ROIs'][rk][rr][hh]
                    rois_dict[rk][rr][hh] = np.array(f[ref]).astype(int)
                    if rk == 'voxel_inds':
                        a = np.array(f[ref]).astype(int)
                        if len(a.shape) == 2:
                            all_vox_list.append(a)

    all_vox_list = np.concatenate(all_vox_list, axis=0)
    assert len(np.unique(all_vox_list)) == len(all_vox_list)
    assert np.all(np.unique(all_vox_list) == samples['all_vox_concat'][:, 0])

    samples['ROIs'] = rois_dict
    samples['ROI_names'] = ROI_names
    samples['hemis'] = hemis

    return samples

def load_main_data(sub_id, make_time_resolved=False, use_bigIPS=True, concat_IPS=True):
    """
    Self-contained version of data_utils.load_main_task_data
    using DATA_ROOT instead of an external module.
    """
    ss = int(sub_id)

    if use_bigIPS:
        sample_fn = os.path.join(DATA_ROOT, 'Samples', f'SampleFile_bigIPS_S{ss:02d}.mat')
    else:
        sample_fn = os.path.join(DATA_ROOT, 'Samples', f'SampleFile_S{ss:02d}.mat')

    samples = load_samplefile_h5py(sample_fn)

    # timing file (GetEventTiming.m)
    timing_fn = os.path.join(DATA_ROOT, 'Samples', f'TimingFile_S{ss:02d}.mat')
    timing = spio.loadmat(timing_fn, squeeze_me=True, struct_as_record=False)
    main = _todict(timing['main'])

    behav_fn = os.path.join(
        DATA_ROOT, 'DataBehavior', f'S{ss:02d}', f'S{ss:02d}_maintask_preproc_all.csv'
    )
    bdat = pd.read_csv(behav_fn, index_col=0)

    if ss <= 7:
        nTRs = 327 - 16
        avgTRs_targ = [4, 7]
        nTRs_concat = 14
    else:
        nTRs = 201
        avgTRs_targ = [2, 5]
        nTRs_concat = 9

    nTrialsPerRun = 48

    roi_names = samples['ROI_names']
    n_rois = len(roi_names)
    n_hemis = len(samples['hemis'])

    main_data = []
    main_data_by_tr = []

    for rr in range(n_rois):
        dat_this_roi = []

        for hh in range(n_hemis):
            roi_inds_num = np.array(samples['ROIs']['voxel_inds'][rr][hh])
            inds_this_hemi = np.isin(np.array(samples['all_vox_concat']), roi_inds_num)[:, 0]

            if np.sum(inds_this_hemi) > 0:
                dat_this_hemi = samples['samplesMain'][inds_this_hemi, :].T
                dat_this_roi.append(dat_this_hemi)

        dat_this_roi = np.concatenate(dat_this_roi, axis=1)
        nVox = dat_this_roi.shape[1]

        nTRsTotal = dat_this_roi.shape[0]
        assert np.mod(nTRsTotal, nTRs) == 0
        nRuns = int(nTRsTotal / nTRs)
        assert nTRsTotal == len(main['RunLabels'])
        assert nRuns == len(np.unique(main['RunLabels']))

        # trial onset indices
        event_diff = np.diff(np.array([0] + list(main['EventLabels'])))
        event_diff_reshaped = np.reshape(event_diff, [nTRs, nRuns], order='F')
        trial_onset_bool = event_diff_reshaped == 1
        trial_onset_bool = np.reshape(trial_onset_bool, [nTRs * nRuns, 1], order='F')
        trial_onset_num = np.where(trial_onset_bool)[0]

        nTrials = nRuns * nTrialsPerRun
        assert len(trial_onset_num) == nTrials

        if rr == 0:
            # alignment checks vs bdat
            resp1 = np.array(main['ResponseLabels'])[trial_onset_num].astype(float)
            resp1[resp1 == 0] = np.nan
            resp2 = np.array(bdat['resp'])
            has_nans = np.isnan(resp1)
            assert np.all(np.isnan(resp2[has_nans]))
            assert np.all(resp1[~has_nans] == resp2[~has_nans])

            vals1 = np.array(main['IsMainLabels'])[trial_onset_num]
            vals2 = np.array(bdat['is_main_grid'])
            assert np.all(vals1 == vals2)

            vals1 = np.array(main['PointLabels'])[:, 0][trial_onset_num]
            vals2 = np.array(bdat['ptx'])
            assert np.all(vals1 == vals2)

            vals1 = np.array(main['PointLabels'])[:, 1][trial_onset_num]
            vals2 = np.array(bdat['pty'])
            assert np.all(vals1 == vals2)

            vals1 = np.array(main['BoundLabels'])[trial_onset_num]
            vals2 = np.array(bdat['task'])
            assert np.all(vals1 == vals2)

            vals1 = np.array(main['MapLabels'])[trial_onset_num]
            vals2 = np.array(bdat['map'])
            assert np.all(vals1 == vals2)

            vals1 = np.array(main['SessLabels'])[trial_onset_num]
            vals2 = np.array(bdat['sess'])
            assert np.all(vals1 == vals2)

            vals1 = np.array(main['RunLabels'])[trial_onset_num]
            vals2 = np.array(bdat['run_overall'])
            assert np.all(vals1 == vals2)

            vals1 = np.array(main['TrialLabels'])[trial_onset_num]
            vals2 = np.array(bdat['trial_overall'])
            assert np.all(vals1 == vals2)

        # zscore per run
        for run in np.unique(main['RunLabels']):
            run_inds = main['RunLabels'] == run
            assert np.sum(run_inds) == nTRs
            dat_this_roi[run_inds, :] = sstats.zscore(dat_this_roi[run_inds, :], axis=0)

        dat_avg_targ = np.full((nTrials, nVox), np.nan)
        if make_time_resolved:
            dat_by_tr = np.full((nTrials, nTRs_concat, nVox), np.nan)
        else:
            dat_by_tr = None

        triCnt = -1

        for run in np.unique(main['RunLabels']):
            run_inds = main['RunLabels'] == run
            assert np.sum(run_inds) == nTRs

            cur_dat = dat_this_roi[run_inds, :]

            these_targ_onsets = np.where(trial_onset_bool[run_inds])[0]
            assert len(these_targ_onsets) == nTrialsPerRun

            for tt in these_targ_onsets:
                window_start = tt + avgTRs_targ[0]
                window_stop  = tt + avgTRs_targ[1]

                if window_stop < nTRs:
                    triCnt += 1
                    dat_avg_targ[triCnt, :] = np.mean(cur_dat[window_start:window_stop, :], axis=0)

                    if make_time_resolved:
                        for tr in range(nTRs_concat):
                            dat_by_tr[triCnt, tr, :] = cur_dat[tt + tr, :]

        assert triCnt == nTrials - 1
        assert not np.any(np.isnan(dat_avg_targ.ravel()))
        if make_time_resolved:
            assert not np.any(np.isnan(dat_by_tr.ravel()))

        main_data.append(dat_avg_targ)
        main_data_by_tr.append(dat_by_tr)

    if concat_IPS:
        ips_roi = [5, 6, 7, 8]
        roi_names = roi_names[0:5] + roi_names[9:] + ['IPSall']

        main_data_new = main_data[0:5] + main_data[9:]
        ips_concat = np.concatenate([main_data[rr] for rr in ips_roi], axis=1)
        main_data_new.append(ips_concat)
        main_data = main_data_new

        if make_time_resolved:
            main_data_by_tr_new = main_data_by_tr[0:5] + main_data_by_tr[9:]
            ips_concat = np.concatenate([main_data_by_tr[rr] for rr in ips_roi], axis=2)
            main_data_by_tr_new.append(ips_concat)
            main_data_by_tr = main_data_by_tr_new

    main_labels = bdat

    # CHANGED: only return 3 items to match callers
    return main_data, main_labels, roi_names

def load_repeat_data(sub_id, make_time_resolved=False, use_bigIPS=True, concat_IPS=True):
    """
    Self-contained version of data_utils.load_repeat_task_data
    using DATA_ROOT instead of an external module.
    """
    ss = int(sub_id)

    if use_bigIPS:
        sample_fn = os.path.join(DATA_ROOT, 'Samples', f'SampleFile_bigIPS_S{ss:02d}.mat')
    else:
        sample_fn = os.path.join(DATA_ROOT, 'Samples', f'SampleFile_S{ss:02d}.mat')

    samples = load_samplefile_h5py(sample_fn)

    timing_fn = os.path.join(DATA_ROOT, 'Samples', f'TimingFile_S{ss:02d}.mat')
    timing = spio.loadmat(timing_fn, squeeze_me=True, struct_as_record=False)
    rep = _todict(timing['rep'])

    behav_fn = os.path.join(
        DATA_ROOT, 'DataBehavior', f'S{ss:02d}', f'S{ss:02d}_reptask_preproc_all.csv'
    )
    bdat = pd.read_csv(behav_fn, index_col=0)

    if ss <= 7:
        nTRs = 329 - 16
        avgTRs_targ = [4, 7]
        nTRs_concat = 14
    else:
        nTRs = 203
        avgTRs_targ = [2, 5]
        nTRs_concat = 9

    nTrialsPerRun = 48

    roi_names = samples['ROI_names']
    n_rois = len(roi_names)
    n_hemis = len(samples['hemis'])

    rep_data = []
    rep_data_by_tr = []

    for rr in range(n_rois):
        dat_this_roi = []

        for hh in range(n_hemis):
            roi_inds_num = np.array(samples['ROIs']['voxel_inds'][rr][hh])
            inds_this_hemi = np.isin(np.array(samples['all_vox_concat']), roi_inds_num)[:, 0]

            if np.sum(inds_this_hemi) > 0:
                dat_this_hemi = samples['samplesRep'][inds_this_hemi, :].T
                dat_this_roi.append(dat_this_hemi)

        dat_this_roi = np.concatenate(dat_this_roi, axis=1)
        nVox = dat_this_roi.shape[1]

        nTRsTotal = dat_this_roi.shape[0]
        assert np.mod(nTRsTotal, nTRs) == 0
        nRuns = int(nTRsTotal / nTRs)
        assert nTRsTotal == len(rep['RunLabels'])
        assert nRuns == len(np.unique(rep['RunLabels']))

        event_diff = np.diff(np.array([0] + list(rep['EventLabels'])))
        event_diff_reshaped = np.reshape(event_diff, [nTRs, nRuns], order='F')
        trial_onset_bool = event_diff_reshaped == 1
        trial_onset_bool = np.reshape(trial_onset_bool, [nTRs * nRuns, 1], order='F')
        trial_onset_num = np.where(trial_onset_bool)[0]

        nTrials = nRuns * nTrialsPerRun
        assert len(trial_onset_num) == nTrials

        if rr == 0:
            resp1 = np.array(rep['ResponseLabels'])[trial_onset_num].astype(float)
            resp1[resp1 == 0] = np.nan
            resp2 = np.array(bdat['resp'])
            has_nans = np.isnan(resp1)
            assert np.all(np.isnan(resp2[has_nans]))
            assert np.all(resp1[~has_nans] == resp2[~has_nans])

            vals1 = np.array(rep['IsMainLabels'])[trial_onset_num]
            vals2 = np.array(bdat['is_main_grid'])
            assert np.all(vals1 == vals2)

            vals1 = np.array(rep['PointLabels'])[:, 0][trial_onset_num]
            vals2 = np.array(bdat['ptx'])
            assert np.all(vals1 == vals2)

            vals1 = np.array(rep['PointLabels'])[:, 1][trial_onset_num]
            vals2 = np.array(bdat['pty'])
            assert np.all(vals1 == vals2)

            vals1 = np.array(rep['MapLabels'])[trial_onset_num]
            vals2 = np.array(bdat['map'])
            assert np.all(vals1 == vals2)

            vals1 = np.array(rep['SessLabels'])[trial_onset_num]
            vals2 = np.array(bdat['sess'])
            assert np.all(vals1 == vals2)

            vals1 = np.array(rep['RunLabels'])[trial_onset_num]
            vals2 = np.array(bdat['run_overall'])
            assert np.all(vals1 == vals2)

            vals1 = np.array(rep['TrialLabels'])[trial_onset_num]
            vals2 = np.array(bdat['trial_overall'])
            assert np.all(vals1 == vals2)

        for run in np.unique(rep['RunLabels']):
            run_inds = rep['RunLabels'] == run
            assert np.sum(run_inds) == nTRs
            dat_this_roi[run_inds, :] = sstats.zscore(dat_this_roi[run_inds, :], axis=0)

        dat_avg_targ = np.full((nTrials, nVox), np.nan)
        if make_time_resolved:
            dat_by_tr = np.full((nTrials, nTRs_concat, nVox), np.nan)
        else:
            dat_by_tr = None

        triCnt = -1

        for run in np.unique(rep['RunLabels']):
            run_inds = rep['RunLabels'] == run
            assert np.sum(run_inds) == nTRs

            cur_dat = dat_this_roi[run_inds, :]

            these_targ_onsets = np.where(trial_onset_bool[run_inds])[0]
            assert len(these_targ_onsets) == nTrialsPerRun

            for tt in these_targ_onsets:
                window_start = tt + avgTRs_targ[0]
                window_stop  = tt + avgTRs_targ[1]

                if window_stop < nTRs:
                    triCnt += 1
                    dat_avg_targ[triCnt, :] = np.mean(cur_dat[window_start:window_stop, :], axis=0)

                    if make_time_resolved:
                        for tr in range(nTRs_concat):
                            dat_by_tr[triCnt, tr, :] = cur_dat[tt + tr, :]

        assert triCnt == nTrials - 1
        assert not np.any(np.isnan(dat_avg_targ.ravel()))
        if make_time_resolved:
            assert not np.any(np.isnan(dat_by_tr.ravel()))

        rep_data.append(dat_avg_targ)
        rep_data_by_tr.append(dat_by_tr)

    if concat_IPS:
        ips_roi = [5, 6, 7, 8]
        roi_names = roi_names[0:5] + roi_names[9:] + ['IPSall']

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
    # CHANGED: only return 3 items to match callers
    return rep_data, rep_labels, roi_names


def rename_ips(roi_names):
    return ["IPS" if r == "IPSall" else r for r in roi_names]

def make_binary_labels(q_arr, boundary_name):
    g0, g1 = boundaries[boundary_name]
    y = np.full(q_arr.shape, -1, dtype=int)
    y[np.isin(q_arr, g0)] = 1
    y[np.isin(q_arr, g1)] = 2
    return y

def make_near_far_masks(dist1, dist2, dist3, boundary_name):
    if boundary_name == 'linear1':
        near = np.isclose(dist1, 0.8)
        far  = np.isclose(dist1, 2.4)
        return near, far
    if boundary_name == 'linear2':
        near = np.isclose(dist2, 0.8)
        far  = np.isclose(dist2, 2.4)
        return near, far
    if boundary_name == 'nonlinear':
        near = np.isclose(dist3, 0.8)
        far  = np.isclose(dist3, 2.4)
        return near, far
    near = np.zeros_like(dist1, dtype=bool)
    far  = np.zeros_like(dist1, dtype=bool)
    return near, far

def decode_within_task_one_roi_replica(data_concat, quad_labs, task_labs, cv_labs,
                                       is_main_grid, task_id, boundary_name):
    tinds = (task_labs == task_id)
    if not np.any(tinds):
        return np.nan

    X = data_concat[tinds, :]
    q = quad_labs[tinds]
    runs = cv_labs[tinds]
    mg = is_main_grid[tinds]

    y = make_binary_labels(q, boundary_name)
    if (y <= 0).any():
        return np.nan

    X = X - X.mean(axis=1, keepdims=True)
    X = np.asarray(X, dtype=np.float64)

    nt = X.shape[0]
    pred = np.full(nt, np.nan)

    c_values = np.logspace(-9, 1, 20)
    logo = LeaveOneGroupOut()

    for cv in np.unique(runs):
        tr = (runs != cv) & mg
        te = (runs == cv)

        if tr.sum() < 4 or np.unique(y[tr]).size < 2 or te.sum() == 0:
            continue

        inner_groups = runs[tr]
        inner_cv = logo.split(X[tr], y[tr], groups=inner_groups)

        clf = LogisticRegressionCV(
            cv=inner_cv,
            Cs=c_values,
            multi_class='multinomial',
            solver='lbfgs',
            penalty='l2',
            n_jobs=-1,
            max_iter=20000,
            tol=1e-3
        )
        clf.fit(X[tr], y[tr])
        pred[te] = clf.predict(X[te])

    if np.isnan(pred).any():
        valid = (~np.isnan(pred)) & mg
        if valid.sum() == 0:
            return np.nan
        return (pred[valid] == y[valid]).mean()

    return (pred[mg] == y[mg]).mean()

def decode_with_subset(data_concat, quad_labs, task_labs, cv_labs,
                       is_main_grid, task_id, boundary_name, subset_mask):
    tinds = (task_labs == task_id)
    if not np.any(tinds):
        return np.nan

    X = data_concat[tinds, :]
    q = quad_labs[tinds]
    runs = cv_labs[tinds]
    mg = is_main_grid[tinds]

    y = make_binary_labels(q, boundary_name)
    if (y <= 0).any():
        return np.nan

    X = X - X.mean(axis=1, keepdims=True)
    X = np.asarray(X, dtype=np.float64)

    pred = np.full(len(X), np.nan)
    c_values = np.logspace(-9, 1, 20)
    logo = LeaveOneGroupOut()

    for cv in np.unique(runs):
        tr = (runs != cv) & mg
        te = (runs == cv)

        if tr.sum() < 4 or np.unique(y[tr]).size < 2 or te.sum() == 0:
            continue

        inner_groups = runs[tr]
        inner_cv = logo.split(X[tr], y[tr], groups=inner_groups)

        clf = LogisticRegressionCV(
            cv=inner_cv,
            Cs=c_values,
            multi_class='multinomial',
            solver='lbfgs',
            penalty='l2',
            n_jobs=-1,
            max_iter=20000,
            tol=1e-3
        )
        clf.fit(X[tr], y[tr])
        pred[te] = clf.predict(X[te])

    subset = subset_mask[tinds] & mg
    valid = (~np.isnan(pred)) & subset

    if valid.sum() == 0:
        return np.nan

    return (pred[valid] == y[valid]).mean()


#%% Init variables

num_participants = 10
sub_ids = [str(i).zfill(2) for i in range(1, num_participants + 1)]

boundaries = {
    "linear1": [[1, 4], [2, 3]],
    "linear2": [[1, 2], [3, 4]],
    "nonlinear": [[1, 3], [2, 4]],
}

task_id_to_name = {1: 'linear1', 2: 'linear2', 3: 'nonlinear', 4: 'repeat'}


#%% one-time near/far sanity check (subject 01)

test_sub = sub_ids[0]
main_rois_chk, main_lab_chk, _ = load_main_data(test_sub)
rep_rois_chk, rep_lab_chk, _ = load_repeat_data(test_sub)
labels_chk = pd.concat([main_lab_chk, rep_lab_chk], axis=0)

is_main_grid_chk = (labels_chk['is_main_grid'].to_numpy().astype(int) == 1)
d1_chk = labels_chk['dist_from_bound1'].to_numpy()
d2_chk = labels_chk['dist_from_bound2'].to_numpy()
d3_chk = labels_chk['dist_from_bound3'].to_numpy()

for bname, (dist1, dist2, dist3) in zip(
    ['linear1','linear2','nonlinear'],
    [(d1_chk,d2_chk,d3_chk), (d1_chk,d2_chk,d3_chk), (d1_chk,d2_chk,d3_chk)]
):
    near_mask_chk, far_mask_chk = make_near_far_masks(dist1, dist2, dist3, bname)
    near_n = (near_mask_chk & is_main_grid_chk).sum()
    far_n  = (far_mask_chk  & is_main_grid_chk).sum()
    print(f"[near/far sanity] sub {test_sub}, {bname}: near={near_n}, far={far_n}")
    
# this is a sanity check! if we have wrangled the data correcrtl, there should 768 near and far trials each for both linear1 and linear2
# bit 1152 near trials vs. 384 far trials for the nonlinear task

#%% subject-wise decoding (near/far + full grid)

def subject_decoding_df_replica(sub_id):
    print(f"[decode] Started decoding for subject {sub_id}")

    main_rois, main_lab, main_roi_names = load_main_data(sub_id)
    rep_rois, rep_lab, rep_roi_names   = load_repeat_data(sub_id)

    roi_names = rename_ips(main_roi_names)
    n_rois = len(roi_names)

    concat_labels = pd.concat([main_lab, rep_lab], axis=0)

    cv_main = main_lab['run_overall'].to_numpy().astype(int)
    cv_rep  = rep_lab['run_overall'].to_numpy().astype(int) + cv_main.max()
    cv_labs = np.concatenate([cv_main, cv_rep], axis=0)

    is_main_grid = (concat_labels['is_main_grid'].to_numpy().astype(int) == 1)
    quad_labs = concat_labels['quadrant'].to_numpy().astype(int)
    task_labs = concat_labels['task'].to_numpy().astype(int)
    task_labs[~np.isin(task_labs, [1, 2, 3])] = 4

    dist1 = concat_labels['dist_from_bound1'].to_numpy()
    dist2 = concat_labels['dist_from_bound2'].to_numpy()
    dist3 = concat_labels['dist_from_bound3'].to_numpy()

    roi_arrays = []
    for ri in range(n_rois):
        Xm = main_rois[ri]
        Xr = rep_rois[ri]
        roi_arrays.append(np.concatenate([Xm, Xr], axis=0))

    rows = []
    for rname, X in zip(roi_names, roi_arrays):
        for task_id in (1, 2, 3):
            for bname in ('linear1', 'linear2', 'nonlinear'):
                near_mask, far_mask = make_near_far_masks(dist1, dist2, dist3, bname)

                acc_near = decode_with_subset(
                    X, quad_labs, task_labs, cv_labs, is_main_grid,
                    task_id, bname, subset_mask=near_mask
                )

                acc_far = decode_with_subset(
                    X, quad_labs, task_labs, cv_labs, is_main_grid,
                    task_id, bname, subset_mask=far_mask
                )

                rows.append([sub_id, rname, task_id_to_name[task_id], bname, 'near', acc_near])
                rows.append([sub_id, rname, task_id_to_name[task_id], bname, 'far', acc_far])

    print(f"[decode] Finished decoding for subject {sub_id}")
    return pd.DataFrame(rows, columns=['sub','ROI','Task','Boundary','Dist','ACC'])

#%% Decode data → NEAR/FAR results + Fig2 copy

acc_rows_fig2 = [subject_decoding_df_replica(s) for s in sub_ids]
acc_long_fig2 = pd.concat(acc_rows_fig2, ignore_index=True)

acc_long_fig2.to_csv(
    os.path.join(stats_output_path, 'binary_withintask_ACC_long_NEARFAR_replica.csv'),
    index=False
)

acc_long_fig2.to_csv(
    os.path.join(stats_output_path, 'binary_withintask_ACC_long_FIG2_replica.csv'),
    index=False
)

#%% Run 3-way ANOVA (NEAR ONLY, linear tasks & boundaries only)

def run_task_boundary_roi_anova(df_long, out_csv, print_table=False):

    df_ = df_long.copy()

    df_ = df_[df_['Task'].isin(['linear1','linear2']) &
              df_['Boundary'].isin(['linear1','linear2']) &
              (df_['Dist'] == 'near')].copy()

    df_ = df_.dropna(subset=['ACC'])

    for col in ['sub','ROI','Task','Boundary']:
        df_[col] = df_[col].astype(str)

    aov = AnovaRM(df_, depvar='ACC', subject='sub',
                  within=['ROI','Task','Boundary']).fit()

    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    aov.anova_table.to_csv(out_csv)

    if print_table:
        print(aov.anova_table)

    return aov

acc_long_nearfar = pd.read_csv(os.path.join(
    stats_output_path, 'binary_withintask_ACC_long_NEARFAR_replica.csv'
))

aov = run_task_boundary_roi_anova(
    acc_long_nearfar,
    out_csv=os.path.join(stats_output_path, 'anova_table_LINEAR_NEARONLY_replica.csv'),
)

anova_result = pd.read_csv(os.path.join(
    stats_output_path, 'anova_table_LINEAR_NEARONLY_replica.csv'
))

# adding partial-eta (reusing the functions from my post-hoc power analysis)
def effect_sizes_from_F(F, df1, df2):
    eta_p2 = (F * df1) / (F * df1 + df2)
    f = math.sqrt(eta_p2 / (1.0 - eta_p2))  # Cohen's f
    return eta_p2, f

eta_vals = []
f_vals = []

for Fi, d1, d2 in zip(anova_result['F Value'], anova_result['Num DF'], anova_result['Den DF']):
    eta_p2, f = effect_sizes_from_F(Fi, d1, d2)
    eta_vals.append(eta_p2)
    f_vals.append(f)

anova_result['eta_p2'] = eta_vals
anova_result['cohens_f'] = f_vals

anova_result.to_csv(os.path.join(
    stats_output_path, 'anova_table_LINEAR_NEARONLY_replica_with_effectsizes.csv'
), index=False)

print(anova_result)

#%% apa-style anova table (image; no cohens f)

def save_apa_anova_table(anova_result, out_png):

    df = anova_result.copy()

    if 'Unnamed: 0' in df.columns:
        df = df.rename(columns={'Unnamed: 0': 'Effect'})
    else:
        df = df.reset_index().rename(columns={'index': 'Effect'})

    def fmt_p(p):
        if pd.isna(p):
            return ''
        if p < 0.001:
            return '< .001'
        return f"{p:.3f}".lstrip('0')

    df['Num DF'] = df['Num DF'].astype(int)
    df['Den DF'] = df['Den DF'].astype(int)
    df['F Value'] = df['F Value'].map(lambda x: f"{x:.2f}")
    df['Pr > F'] = df['Pr > F'].map(fmt_p)
    df['eta_p2'] = df['eta_p2'].map(lambda x: f"{x:.02f}")

    df = df.rename(columns={
        'Num DF': 'df1',
        'Den DF': 'df2',
        'F Value': 'F',
        'Pr > F': 'p',
        'eta_p2': 'np²'
    })
    df = df[['Effect','df1','df2','F','p','np²']]

    n_rows = len(df)
    fig, ax = plt.subplots(figsize=(6, 0.6 + 0.3 * n_rows))
    ax.axis('off')

    table = ax.table(
        cellText=df.values,
        colLabels=df.columns,
        loc='center',
        cellLoc='right'
    )

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    for col in range(len(df.columns)):
        table.auto_set_column_width(col)

    n_cols = len(df.columns)
    for (r, c), cell in table.get_celld().items():
        cell.visible_edges = ''
        cell.set_height(0.12)
        if r == 0:
            cell.set_text_props(weight='bold')
            cell.set_facecolor('#F2F2F2')
        else:
            cell.set_facecolor('white')

        if c == 0:
            cell._loc = 'left'
            cell._text.set_ha('left')
        else:
            cell._loc = 'right'
            cell._text.set_ha('right')

    left = table[(0, 0)].get_x()
    right = table[(0, n_cols - 1)].get_x() + table[(0, n_cols - 1)].get_width()

    top_header = table[(0, 0)].get_y() + table[(0, 0)].get_height()
    bottom_header = table[(0, 0)].get_y()
    bottom_table = table[(n_rows, 0)].get_y()

    ax.hlines(top_header,   left, right, linewidth=1.5, color='black')
    ax.hlines(bottom_header,left, right, linewidth=1.0, color='black')
    ax.hlines(bottom_table, left, right, linewidth=1.5, color='black')

    fig.tight_layout()
    fig.savefig(out_png, dpi=600, bbox_inches='tight')
    plt.close(fig)


save_apa_anova_table(
    anova_result,
    out_png=os.path.join(fig_outdir, 'Table1_LINEAR_NEARONLY_replica_APA.png')
)

#%% Figure 2A–C plots

acc_long_fig2 = pd.read_csv(os.path.join(
    stats_output_path, 'binary_withintask_ACC_long_FIG2_replica.csv'
))

roi_order = ['V1','V2','V3','V3AB','hV4','LO1','LO2','IPS']
task_order = ['linear1','linear2','nonlinear']
task_labels = {
    'linear1':'Linear-1 Task',
    'linear2':'Linear-2 Task',
    'nonlinear':'Nonlinear Task'
}
task_colors = {
    'linear1':'#1f4e79',
    'linear2':'#2e86c1',
    'nonlinear':'#76d7c4'
}
chance_y = 0.5

offsets = {'linear1': -0.18, 'linear2': 0.0, 'nonlinear': +0.18}

panels = [
    ('linear1', 'A  Binary classifier: Predict "Linear-1" category'),
    ('linear2', 'B  Binary classifier: Predict "Linear-2" category'),
    ('nonlinear', 'C  Binary classifier: Predict "Nonlinear" category')
]

def plot_panel(boundary, title):
    df_b = acc_long_fig2.query("Boundary == @boundary and Task in @task_order").copy()

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.axhline(chance_y, color='gray', linewidth=1.0)

    x_base = np.arange(len(roi_order))

    for task in task_order:
        df_t = df_b[df_b['Task'] == task]

        for s, df_s in df_t.groupby('sub'):
            df_s_roi = df_s.groupby('ROI', as_index=True)['ACC'].mean()
            y = df_s_roi.reindex(roi_order).values
            x = x_base + offsets[task]
            ax.scatter(x, y, s=12, color='#BFBFBF', zorder=1)

        g = df_t.groupby('ROI')['ACC']
        means = g.mean().reindex(roi_order).values
        errors = g.apply(lambda x: sem(x, nan_policy='omit')).reindex(roi_order).values
        x = x_base + offsets[task]

        ax.errorbar(
            x, means, yerr=errors,
            fmt='o', markersize=8, capsize=3, linewidth=1.2,
            color=task_colors[task], ecolor=task_colors[task],
            label=task_labels[task], zorder=3
        )

    ax.set_xticks(x_base)
    ax.set_xticklabels(roi_order, fontsize=18)
    ax.set_ylim(0.4, 1.0)
    ax.tick_params(axis='y', labelsize=18)
    ax.set_ylabel('Classifier accuracy', fontsize=16)
    ax.set_title(title, fontsize=18)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.legend(frameon=False, loc='upper right')
    fig.tight_layout()

    return fig, ax

for i, (bnd, ttl) in enumerate(panels, 1):
    fig, ax = plot_panel(bnd, ttl)
    out_png = os.path.join(fig_outdir, f"Figure2_{bnd}_replica_DOTSEM.png")
    fig.savefig(out_png, dpi=300, bbox_inches='tight')
    plt.show()
