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
import h5py as _h5 # for looking at mat files

# actual stats packages
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import StratifiedKFold
from statsmodels.stats.anova import AnovaRM
import h5py # for converting mats
from scipy.io import loadmat
from scipy.stats import zscore
from sklearn.feature_selection import f_classif

# for plotting
import matplotlib.pyplot as plt
from scipy.stats import sem

np.random.seed(0) # for reproducability

#%% Init variables + set paths

# set paths (I used os package since it's platform independent -- easier for someone to reuse)
root_dir = '/Users/lenakemmelmeier/Documents/GitHub/henderson2025' # root_dir for the repo
bold_data_path = os.path.join(root_dir, 'data', 'Samples') # path to save/read npz files (kept with original mats)
stats_output_path = os.path.join(root_dir, 'bold_decoding_anova_results') # output path for stats CSVs
sys.path.append(root_dir) # add the repo to the path

num_participants = 10 # number of participants

# uses list comprehension to make a list of sub IDs to iterate over later
sub_ids = [str(i).zfill(2) for i in range(1, num_participants + 1)] # zero pads that sub IDs up to 2 digits (e.g., '1' becomes '01')

boundaries = {"linear1": [[1, 4], [2, 3]], "linear2": [[1, 2], [3, 4]], "nonlinear": [[1, 3], [2, 4]]} # dict for which quartiles correspond to which boundary groupings in the different conditions

#%% Helper functions (run this before the cells below!)

def _zscore_within_runs(X, run_labels):
    
    for r in np.unique(run_labels):
        idx = (run_labels == r)
        X[idx, :] = zscore(X[idx, :], axis=0, nan_policy="omit")
        
    return X

def _build_task(dat_TV, run_labels, targ_on_bool, roi_vox_ids, voxel_id_to_col, avg_start, avg_stop):
    
    dat_TV = _zscore_within_runs(dat_TV, run_labels)

    # collect column indices per ROI (merge hemispheres)
    roi_cols = []
    for ri in range(len(roi_vox_ids)):
        cols = []
        
        for hi in range(len(roi_vox_ids[ri])):
            
            for v in roi_vox_ids[ri][hi]:
                v = int(v)
                
                if v in voxel_id_to_col:
                    cols.append(voxel_id_to_col[v])
                    
        roi_cols.append(np.array(cols, dtype=int))

    # average window [avg_start, avg_stop) after each onset per run
    all_roi_arrays = []
    unique_runs = np.unique(run_labels)
    
    for ri in range(len(roi_cols)):
        cols = roi_cols[ri]
        A = []
        
        for r in unique_runs:
            ridx = (run_labels == r)
            Xr = dat_TV[ridx][:, cols]
            onsets = np.where(targ_on_bool[ridx])[0]
            
            for t0 in onsets:
                t1, t2 = t0 + avg_start, t0 + avg_stop
                
                if t2 <= Xr.shape[0]:
                    A.append(Xr[t1:t2, :].mean(axis=0))
                    
        all_roi_arrays.append(np.vstack(A) if len(A) else np.empty((0, len(cols))))
        
    return all_roi_arrays


def get_run_labels_safe(f, node, T_expected):
    obj = f[node]
    
    if isinstance(obj, _h5.Dataset):
        return np.ones((T_expected,), dtype=int)
    
    def _ok(ds):
        return ds.ndim == 1 and ds.shape[0] == T_expected and ds.dtype.kind in ('i','u','f')
    
    for _, child in obj.items():
        
        if isinstance(child, _h5.Dataset) and _ok(child):
            
            arr = np.array(child[:]).ravel()
            
            if arr.dtype.kind == 'f':
                arr = arr.round().astype(int)
            if arr.min() < 1:
                arr = (arr - arr.min() + 1).astype(int)
            return arr
        
    return np.ones((T_expected,), dtype=int)


def ensure_time_by_voxel(arr, nvox_expected, label):
    
    if arr.ndim != 2:
        return arr
    if arr.shape[1] == nvox_expected:
        return arr
    if arr.shape[0] == nvox_expected:
        return arr.T
    
    return arr

# define a function to convert the mat file data to npz (python-compatible version)
def convert_mat_to_npz(sub_ids, repo_root, out_dir):

    os.makedirs(out_dir, exist_ok=True)

    # loop over participants!
    for sub_id in sub_ids:
        
        # get the paths to the timing/bold signal files
        sample_file = os.path.join(repo_root, "data", "Samples", f"SampleFile_bigIPS_S{int(sub_id):02d}.mat")
        timing_file = os.path.join(repo_root, "data", "Samples", f"TimingFile_S{int(sub_id):02d}.mat")

        # TR window (Trio S01–S07 vs Prisma S08+)
        if int(sub_id) <= 7:
            avg_start, avg_stop = 4, 7
        else:
            avg_start, avg_stop = 2, 5

        with h5py.File(sample_file, "r") as f:
            # voxel ID → column
            all_vox_concat = np.array(f["/all_vox_concat"][:], dtype=int).ravel()
            voxel_id_to_col = {int(v): i for i, v in enumerate(all_vox_concat)}
            nvox = all_vox_concat.size

            # ROI voxel indices (object refs) at /ROIs/voxel_inds
            ROIs = f["/ROIs/voxel_inds"]
            n_rois = ROIs.shape[0]
            n_hemi = ROIs.shape[1] if len(ROIs.shape) > 1 else 1
            roi_names = [f"ROI_{i+1}" for i in range(n_rois)]  # no names present in file
            roi_vox_ids = []
            
            for ri in range(n_rois):
                per_hemi = []
                
                for hi in range(n_hemi):
                    ref = ROIs[ri][hi] if n_hemi > 1 else ROIs[ri]
                    per_hemi.append(np.array(f[ref][:], dtype=int).ravel())
                roi_vox_ids.append(per_hemi)

            # main/rep as datasets at root; coerce to T×Nvox
            data_main = ensure_time_by_voxel(np.array(f["/samplesMain"][:]), nvox, "samplesMain")
            data_rep = ensure_time_by_voxel(np.array(f["/samplesRep"][:]), nvox, "samplesRep")

            # run labels (fallback = ones)
            run_main = get_run_labels_safe(f, "/samplesMain", data_main.shape[0])
            run_rep = get_run_labels_safe(f, "/samplesRep", data_rep.shape[0])

        # loading the timing file
        tm = loadmat(timing_file, squeeze_me=True, struct_as_record=False)

        main_struct = tm['main']
        rep_struct  = tm['rep']
        
        # build onset booleans from EventLabels + RunLabels (per-run rising edges)
        targ_on_main = build_trial_onset_bool(getattr(main_struct, 'EventLabels'), getattr(main_struct, 'RunLabels'))
        targ_on_rep = build_trial_onset_bool(getattr(rep_struct, 'EventLabels'), getattr(rep_struct, 'RunLabels'))

        # build per-ROI trial-averaged arrays
        main_rois = _build_task(data_main, run_main, targ_on_main, roi_vox_ids, voxel_id_to_col, avg_start, avg_stop)
        rep_rois  = _build_task(data_rep,  run_rep,  targ_on_rep, roi_vox_ids, voxel_id_to_col, avg_start, avg_stop)

        # (No IPS0–IPS3 names in these files) → keep generic ROI_1..ROI_N
        roi_names_out = roi_names
        main_out = main_rois
        rep_out  = rep_rois

        # write .npz
        np.savez(os.path.join(out_dir, f"S{sub_id}_main_data.npz"), roi_names=np.array(roi_names_out, dtype=object), **{roi_names_out[i]: main_out[i] for i in range(len(roi_names_out))})
        np.savez(os.path.join(out_dir, f"S{sub_id}_repeat_data.npz"), roi_names=np.array(roi_names_out, dtype=object), **{roi_names_out[i]: rep_out[i] for i in range(len(roi_names_out))})

# defines a function to load the bold data
def load_bold_data(sub_id, data_path, task_type="main"):
    
    # read in the labels from the behavioral data
    if task_type == "repeat":
        filename = f"S{sub_id}_reptask_preproc_all.csv"
    else:
        filename = f"S{sub_id}_maintask_preproc_all.csv"
    
    behavioral_data_path = os.path.join(root_dir, 'data', f"DataBehavior/S{sub_id}", filename)

    labels = pd.read_csv(behavioral_data_path, index_col=0)
    
    # get the bold data (we converted this from a mat file to a npz file earlier)
    npz_path = os.path.join(data_path, f"S{sub_id.zfill(2)}_{task_type}_data.npz")

    # npz should contain: roi_names (list of string vals) and arrays keyed by roi name
    packed = np.load(npz_path, allow_pickle=True)
    roi_names = list(packed["roi_names"])
    data_list = [packed[roi] for roi in roi_names]

    return data_list, labels, roi_names


# makes y for a given boundary (maps quadrant -> 0/1)
def make_binary_labels(q_arr, boundary_name, boundaries):

    g0, g1 = boundaries[boundary_name]
    y = np.full(q_arr.shape, -1, dtype=int)
    y[np.isin(q_arr, g0)] = 0
    y[np.isin(q_arr, g1)] = 1
    
    return y

# leave-one-run-out decoding within task, per ROI
def decode_within_task_one_roi(X, y, runs):
    
    clf = LogisticRegressionCV(Cs=10, cv=3, penalty='l2', solver='liblinear', max_iter=200, n_jobs=1, class_weight='balanced')
    accs = []
    uniq_runs = np.unique(runs)

    if uniq_runs.size >= 2:
        for r in uniq_runs:
            train_mask = (runs != r)
            test_mask  = (runs == r)

            X_train, y_train = X[train_mask],y[train_mask]
            X_test, y_test = X[test_mask], y[test_mask]

            # skip folds with only one class
            if np.unique(y_train).size < 2:
                continue

            clf.fit(X_train, y_train)
            accs.append((clf.predict(X_test) == y_test).mean())
    else:
        
        # single-run fallback
        n_min = min((y == 0).sum(), (y == 1).sum())
        if n_min < 2:
            return np.nan
        
        k = min(5, n_min)
        skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=0)
        
        for tr_idx, te_idx in skf.split(X, y):
            clf.fit(X[tr_idx], y[tr_idx])
            accs.append((clf.predict(X[te_idx]) == y[te_idx]).mean())

    return float(np.mean(accs)) if accs else np.nan

def subject_decoding_df(sub_id, data_path, boundaries, top_k_voxels='min'):

    data_list, labels, roi_names = load_bold_data(sub_id, data_path, task_type="main")
    lbl = labels.copy()
    
    print(f"[decode] Starting subject {sub_id} ...")

    # trial indexing: correct for one-indexed vs. zero-indexed differences between python/matlab
    if 'trial_overall' in lbl.columns:
        keep_idx = lbl['trial_overall'].to_numpy().astype(int) - 1
    else:
        keep_idx = np.arange(len(lbl), dtype=int)

    roi_arrays_main = [arr[keep_idx, :] for arr in data_list]
    roi_arrays_main = [X - X.mean(axis=1, keepdims=True) for X in roi_arrays_main]
    quadrants_main = lbl['quadrant'].to_numpy().astype(int)
    tasks_main = lbl['task'].to_numpy().astype(int)
    runs_main = lbl['run_overall'].to_numpy().astype(int)

    data_list_rep, labels_rep, _ = load_bold_data(sub_id, data_path, task_type="repeat")
    lbl_rep = labels_rep.copy()

    # restrict to main-grid trials
    if 'is_main_grid' in lbl_rep.columns:
        rep_mask = lbl_rep['is_main_grid'].to_numpy().astype(bool)
    else:
        rep_mask = np.ones(len(lbl_rep), dtype=bool)

    quadrants_rep = lbl_rep['quadrant'].to_numpy().astype(int)[rep_mask]
    data_list_rep = [X[rep_mask, :] for X in data_list_rep]

    # determine top-k voxels per ROI (equalized across ROIs)
    nvox_per_roi = [arr.shape[1] for arr in data_list_rep]
    top_k = int(np.min(nvox_per_roi)) if top_k_voxels == 'min' else int(top_k_voxels)

    # run ANOVA voxel selection on repeat task
    voxel_masks = []
    for X_rep in data_list_rep:
        mask = select_voxels_by_quadrant_anova(X_rep, quadrants_rep, top_k)
        voxel_masks.append(mask)
        
    print(f"[decode] Finished voxel selection for subject {sub_id}")

    # apply voxel masks to main-task data
    roi_arrays = [X[:, mask] for X, mask in zip(roi_arrays_main, voxel_masks)]

    # decoding!!
    task_id_to_name = {1: 'linear1', 2: 'linear2', 3: 'nonlinear'}
    rows = []
    
    for task_id in (1, 2):
        tmask = (tasks_main == task_id)
        if tmask.sum() == 0:
            continue
        q_t = quadrants_main[tmask]
        runs_t = runs_main[tmask]

        for bname in ('linear1', 'linear2'):
            y = make_binary_labels(q_t, bname, boundaries)
            valid = (y >= 0)
            y = y[valid]; runs_tb = runs_t[valid]

            for roi_name, X_roi in zip(roi_names, roi_arrays):
                X_t = X_roi[tmask][valid]
                acc = decode_within_task_one_roi(X_t, y, runs_tb)
                rows.append([sub_id, roi_name, task_id_to_name[task_id], bname, acc])
                
    print(f"[decode] Finished decoding for subject {sub_id}")

    return pd.DataFrame(rows, columns=['sub','ROI','Task','Boundary','ACC'])


# runs a three-way ANOVA w/ ROI x Task x Boundary (only for linear tasks/boundaries)
def run_task_boundary_roi_anova(df_long, out_csv, print_table):

    df_ = df_long.copy()

    # keep only linear tasks/boundaries
    df_ = df_[df_['Task'].isin(['linear1','linear2']) & df_['Boundary'].isin(['linear1','linear2'])].copy()
    df_ = df_.dropna(subset=['ACC']) # get rid of NaNs

    # cast to strings for AnovaRM
    for col in ['sub','ROI','Task','Boundary']:
        df_[col] = df_[col].astype(str)

    # fit 3-way RM ANOVA
    aov = AnovaRM(df_, depvar='ACC', subject='sub', within=['ROI','Task','Boundary']).fit()

    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    aov.anova_table.to_csv(out_csv)

    print("\n=== Full 3-way ANOVA (ROI × Task × Boundary) ===")
    print(aov.anova_table)
    print("\n--- Task × Boundary interaction ---")
    print(aov.anova_table.loc[aov.anova_table.index.str.contains('Task:Boundary')])

    return aov

# make a bool vector; signal trial onsets
def build_trial_onset_bool(event_labels, run_labels):

    ev = np.asarray(event_labels).astype(int).ravel()
    runs = np.asarray(run_labels).ravel()

    T = ev.shape[0]

    onset_bool = np.zeros(T, dtype=bool)
    uniq_runs = np.unique(runs)
    
    # iterate each run, do a within-run diff
    for r in uniq_runs:
        
        idx = np.flatnonzero(runs == r)
        
        if idx.size == 0:
            continue
        
        ev_r = ev[idx]
        onset_r = np.diff(np.r_[0, ev_r]) == 1  # rising edge = trial start
        onset_bool[idx] = onset_r
        
    return onset_bool

# one-way ANOVA voxel selection (as in Henderson et al.)
def select_voxels_by_quadrant_anova(trial_by_voxel, quadrants, top_k):

    if trial_by_voxel.shape[0] <= 1:
        return np.ones(trial_by_voxel.shape[1], dtype=bool)
    
    F, _ = f_classif(trial_by_voxel, quadrants)
    F[np.isnan(F)] = 0
    order = np.argsort(F)[::-1]
    keep = order[:min(top_k, len(order))]
    mask = np.zeros(trial_by_voxel.shape[1], dtype=bool)
    mask[keep] = True
    
    return mask

#%% Make npz files once (writes into Samples/)

convert_mat_to_npz(sub_ids, repo_root=root_dir, out_dir=bold_data_path)  # run once, then comment out

#%% Decode each subject
all_rows = []

for s in sub_ids:
    
    df_s = subject_decoding_df(s, bold_data_path, boundaries)
    all_rows.append(df_s)
    
acc_long = pd.concat(all_rows, ignore_index=True)

os.makedirs(stats_output_path, exist_ok=True)
acc_long.to_csv(os.path.join(stats_output_path, 'binary_within_task_acc_long.csv'), index=False)

#%% Run the single target ANOVA (ROI × Task × Boundary), report Task×Boundary
aov = run_task_boundary_roi_anova(acc_long, out_csv=os.path.join(stats_output_path, 'anova_table.csv'))

#%% Figure 2A–C style plots (I copied the authors' style here)

fig_outdir = os.path.join(stats_output_path, "figs") # ourpur path for figs
os.makedirs(fig_outdir, exist_ok=True)

boundaries_present = [b for b in ["linear1", "linear2", "nonlinear"] if b in acc_long["Boundary"].unique().tolist()]
if not boundaries_present:
    print("No boundaries found to plot.")
    
else:
    def ordered_rois(df):
        
        rois = list(df["ROI"].unique())
        
        if "IPS" in rois:
            rois = [r for r in rois if r != "IPS"] + ["IPS"]
            
        return rois

    # Colors by Task (consistent across panels)
    task_levels = ["linear1","linear2"]
    task_colors = {"linear1":"#4C78A8", "linear2":"#F58518"}  # blue/orange-ish; adjust as needed
    bar_width = 0.36
    jitter = 0.06

    # One row with len(boundaries_present) panels
    n_pan = len(boundaries_present)
    fig, axes = plt.subplots(1, n_pan, figsize=(6.0*n_pan, 5.0), constrained_layout=True)

    if n_pan == 1:
        axes = [axes]

    for ax, bnd in zip(axes, boundaries_present):
        df_b = acc_long[acc_long["Boundary"] == bnd].copy()

        # keep only Tasks we have (usually ['linear1','linear2'])
        tasks_here = [t for t in task_levels if t in df_b["Task"].unique().tolist()]
        rois = ordered_rois(df_b)

        # get mean and sem across subjects for each ROI × Task
        means = {}
        errors = {}
        for t in tasks_here:
            subgrp = df_b[df_b["Task"] == t].groupby("ROI")["ACC"]
            means[t]  = subgrp.mean().reindex(rois)
            errors[t] = subgrp.apply(lambda x: sem(x, nan_policy="omit")).reindex(rois)

        x = np.arange(len(rois))
        # Bars
        for i, t in enumerate(tasks_here):
            offs = (-0.5 + i) * bar_width
            ax.bar(x + offs, means[t].values, width=bar_width, yerr=errors[t].values, capsize=3, linewidth=1.0, edgecolor="black", color=task_colors.get(t, None), alpha=0.9, label=t)

        # Per-subject dots with light lines (paired within ROI)
        # Pivot to sub × (ROI, Task)
        for ri, roi in enumerate(rois):
            
            # gather per-subject accuracies for each task at this ROI
            per_task = []
            
            for t in tasks_here:
                tmp = df_b[(df_b["ROI"] == roi) & (df_b["Task"] == t)][["sub","ACC"]].copy()
                tmp = tmp.rename(columns={"ACC": f"ACC_{t}"})
                per_task.append(tmp)
                
            if not per_task:
                continue
            
            M = per_task[0]
            
            for k in range(1, len(per_task)):
                M = M.merge(per_task[k], on="sub", how="outer")

            # plot pairs
            x_positions = [x[ri] + (-0.5 + i)*bar_width for i in range(len(tasks_here))]
            
            for _, row in M.iterrows():
                
                vals = [row.get(f"ACC_{t}", np.nan) for t in tasks_here]
                # light connecting line if both present
                
                if len(vals) == 2 and np.isfinite(vals[0]) and np.isfinite(vals[1]):
                    ax.plot(x_positions, vals, lw=0.6, alpha=0.4, color="gray")
                    
                # dots
                for i, v in enumerate(vals):
                    
                    if np.isfinite(v):
                        
                        ax.scatter(x_positions[i] + np.random.uniform(-jitter, jitter), v, s=18, color="black", alpha=0.7, zorder=5)

        # draw chance line
        ax.axhline(0.5, color="k", lw=1.0, ls="--", alpha=0.6)

        # axes labels, legend, etc.
        ax.set_xticks(x)
        ax.set_xticklabels(rois, rotation=0)
        ax.set_ylim(0.4, 0.8)  # adjust if your data differ; authors' was around .5–.7
        title_map = {"linear1": "A  Linear 1 boundary", "linear2": "B  Linear 2 boundary", "nonlinear": "C  Nonlinear boundary"}
        ax.set_title(title_map.get(bnd, bnd), fontsize=14)
        ax.set_ylabel("Decoding accuracy")
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)

    # make legend
    handles, labels = axes[0].get_legend_handles_labels()
    
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=len(task_levels), frameon=False)

    savepath = os.path.join(fig_outdir, "Figure2ABC_style_avgacc_binary.png")
    fig.savefig(savepath, dpi=300)
    print(f"Saved: {savepath}")