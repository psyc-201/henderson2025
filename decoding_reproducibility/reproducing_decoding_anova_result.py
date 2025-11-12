#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  3 14:43:25 2025

@author: lenakemmelmeier
"""

#%% Set up environment - load in appropiate packages

import numpy as np
import os  # used for configuring paths
import sys # used for configuring paths
import pandas as pd
from pathlib import Path  # for infering path

# actual stats packages
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.linear_model import LogisticRegressionCV
from statsmodels.stats.anova import AnovaRM

# for plotting
from scipy.stats import sem
# author plotting helpers (matches Fig 2 bar/dot style)

import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)

np.random.seed(0)  # for reproducibility

# convergence controls (keep authors' pipeline, just make solver more forgiving)
MAX_ITER = 20000
TOL = 1e-3

#%% Set paths, import helpers from the authors'

# root_dir = '/Users/lenakemmelmeier/Documents/GitHub/henderson2025'
root_dir = Path(__file__).resolve().parents[1]  # infer the root based on where this script is placed
os.chdir(root_dir)

package_dir = os.path.join(root_dir, 'decoding_reproducibility')  # the parent folder containing code_utils (downloaded from authors)

# outputs
stats_output_path = os.path.join(root_dir, 'bold_decoding_anova_results')
os.makedirs(stats_output_path, exist_ok=True)

# make repo importable
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

# add these helpers to path
if os.path.isdir(os.path.join(package_dir, 'code_utils')):
    if package_dir not in sys.path:
        sys.path.insert(0, package_dir)

# now these work because the parent of code_utils/ is on sys.path
from code_utils import data_utils
data_root = os.path.join(root_dir, 'data')
data_utils.root = data_root  # match authors’ expectation for data root

# plot utils lives next to data_util
from code_utils import plot_utils as _plot_utils
set_all_font_sizes = _plot_utils.set_all_font_sizes
plot_multi_bars = _plot_utils.plot_multi_bars

#%% Init other variables

num_participants = 10  # number of participants
sub_ids = [str(i).zfill(2) for i in range(1, num_participants + 1)]  # zero-pad sub IDs to 2 digits

# dict for which quadrants correspond to which boundary groupings in the different conditions (authors' convention)
boundaries = {
    "linear1": [[1, 4], [2, 3]],
    "linear2": [[1, 2], [3, 4]],
    "nonlinear": [[1, 3], [2, 4]],
}

task_id_to_name = {1: 'linear1', 2: 'linear2', 3: 'nonlinear', 4: 'repeat'}

#%% Helper functions (run this before the sections below!)

# call the authors' helper to load the main task trial data (already run-zscored & trial-averaged by authors)
def load_main_data(sub_id):
    ss = int(sub_id)
    main_data, main_by_tr, main_labels, roi_names = data_utils.load_main_task_data(
        ss, make_time_resolved=False, use_bigIPS=True, concat_IPS=True
    )
    return main_data, main_labels, roi_names

# call the authors' helper to load the repeat task trial data
def load_repeat_data(sub_id):
    ss = int(sub_id)
    rep_data, rep_by_tr, rep_labels, roi_names = data_utils.load_repeat_task_data(
        ss, make_time_resolved=False, use_bigIPS=True, concat_IPS=True
    )
    return rep_data, rep_labels, roi_names

# rename IPSall to IPS so plotting code is same as authors'
def rename_ips(roi_names):
    return ["IPS" if r == "IPSall" else r for r in roi_names]

# convert quadrant labels to {1,2} class labels for the requested boundary (authors use {1,2}, not {0,1})
def make_binary_labels(q_arr, boundary_name):
    g0, g1 = boundaries[boundary_name]
    y = np.full(q_arr.shape, -1, dtype=int)
    y[np.isin(q_arr, g0)] = 1
    y[np.isin(q_arr, g1)] = 2
    return y

#%% Direct-replication decoding core (matches authors’ logic exactly)

def decode_within_task_one_roi_replica(data_concat, quad_labs, task_labs, cv_labs, is_main_grid,
                                       task_id, boundary_name):
    # subset to the current task (1,2,3; task 4 = repeat, not decoded for Fig 2)
    tinds = (task_labs == task_id)
    if not np.any(tinds):
        return np.nan

    # slice arrays to current task
    X = data_concat[tinds, :]
    q = quad_labs[tinds]
    runs = cv_labs[tinds]
    mg = is_main_grid[tinds]

    # map quadrants -> {1,2} for the boundary
    y = make_binary_labels(q, boundary_name)
    if (y <= 0).any():
        return np.nan

    # per-trial mean-centering across voxels (authors subtract mean voxel response on each trial)
    X = X - X.mean(axis=1, keepdims=True)
    X = np.asarray(X, dtype=np.float64)  # help BLAS/LBFGS convergence on some builds

    # storage for per-trial predictions across outer folds
    nt = X.shape[0]
    pred = np.full(nt, np.nan)

    # authors’ C grid for LR; multinomial with lbfgs, L2
    c_values = np.logspace(-9, 1, 20)
    logo = LeaveOneGroupOut()

    # outer CV: leave-one-run-out by runs within the current task
    for cv in np.unique(runs):
        tr = (runs != cv) & mg          # TRAIN on main-grid trials only
        te = (runs == cv)               # TEST on held-out run (we will score on its main-grid trials)

        # need enough trials + both classes in training
        if tr.sum() < 4 or np.unique(y[tr]).size < 2 or te.sum() == 0:
            continue

        # inner CV: LOGO on the training runs only (groups = run ids)
        inner_groups = runs[tr]
        inner_cv = logo.split(X[tr], y[tr], groups=inner_groups)

        clf = LogisticRegressionCV(
            cv=inner_cv,
            Cs=c_values,
            multi_class='multinomial',
            solver='lbfgs',
            penalty='l2',
            n_jobs=-1,
            max_iter=MAX_ITER,  # increased iter budget
            tol=TOL             # slightly looser tolerance
        )
        clf.fit(X[tr], y[tr])

        # predict the held-out run
        pred[te] = clf.predict(X[te])

    # final accuracy is computed on MAIN-GRID trials only (to match paper)
    if np.isnan(pred).any():
        valid = (~np.isnan(pred)) & mg
        if valid.sum() == 0:
            return np.nan
        return (pred[valid] == y[valid]).mean()

    return (pred[mg] == y[mg]).mean()

def subject_decoding_df_replica(sub_id):
    print(f"[decode] Started decoding for subject {sub_id}")

    # load main and repeat data
    main_rois, main_lab, main_roi_names = load_main_data(sub_id)
    rep_rois,  rep_lab,  rep_roi_names  = load_repeat_data(sub_id)

    # keep ROI name list consistent with authors' plotting
    roi_names = rename_ips(main_roi_names)
    n_rois = len(roi_names)

    # concatenate labels (MAIN on top of REPEAT)
    concat_labels = pd.concat([main_lab, rep_lab], axis=0)

    # leave-one-run-out groups: offset REPEAT runs so they don't collide with MAIN
    cv_main = main_lab['run_overall'].to_numpy().astype(int)
    cv_rep  = rep_lab['run_overall'].to_numpy().astype(int) + cv_main.max()
    cv_labs = np.concatenate([cv_main, cv_rep], axis=0)

    # label vectors (+ remap non-1/2/3 tasks to 4 = repeat)
    is_main_grid = (concat_labels['is_main_grid'].to_numpy().astype(int) == 1)
    quad_labs    = concat_labels['quadrant'].to_numpy().astype(int)
    task_labs    = concat_labels['task'].to_numpy().astype(int)
    task_labs[~np.isin(task_labs, [1, 2, 3])] = 4

    # build ROI matrices by row-concatenating MAIN then REPEAT
    roi_arrays = []
    for ri in range(n_rois):
        Xm = main_rois[ri]  # [nTrials_main x nVox]
        Xr = rep_rois[ri]   # [nTrials_rep  x nVox]
        roi_arrays.append(np.concatenate([Xm, Xr], axis=0))

    # decode per ROI × task (1/2/3) × boundary
    rows = []
    for rname, X in zip(roi_names, roi_arrays):
        for task_id in (1, 2, 3):  # three categorization tasks for Fig 2
            for bname in ('linear1', 'linear2', 'nonlinear'):
                acc = decode_within_task_one_roi_replica(
                    X, quad_labs, task_labs, cv_labs, is_main_grid,
                    task_id, bname
                )
                rows.append([sub_id, rname, task_id_to_name[task_id], bname, acc])

    print(f"[decode] Finished decoding for subject {sub_id}")
    return pd.DataFrame(rows, columns=['sub','ROI','Task','Boundary','ACC'])

#%% Decode each subject for Fig 2 (no near/far split)
# This reproduces the within-task binary decoding used in Fig 2A–C (authors' logic).
acc_rows_fig2 = [subject_decoding_df_replica(s) for s in sub_ids]
acc_long_fig2 = pd.concat(acc_rows_fig2, ignore_index=True)
acc_long_fig2.to_csv(os.path.join(stats_output_path, 'binary_withintask_ACC_long_FIG2_replica.csv'), index=False)

#%% Run the 3-way ANOVA (ROI × Task × Boundary) on linear-only data (like authors’ table)
def run_task_boundary_roi_anova(df_long, out_csv, print_table=False):

    df_ = df_long.copy()

    # keep only linear tasks/boundaries
    df_ = df_[df_['Task'].isin(['linear1','linear2']) & df_['Boundary'].isin(['linear1','linear2'])].copy()
    df_ = df_.dropna(subset=['ACC'])  # in case any ROI/task/boundary produced NaN (e.g., too few trials/classes)

    # cast to strings for AnovaRM
    for col in ['sub','ROI','Task','Boundary']:
        df_[col] = df_[col].astype(str)

    # fit 3-way repeated-measures ANOVA
    aov = AnovaRM(df_, depvar='ACC', subject='sub', within=['ROI','Task','Boundary']).fit()

    # save the ANOVA table
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    aov.anova_table.to_csv(out_csv)

    if print_table:
        print("\n=== Full 3-way ANOVA (ROI × Task × Boundary) ===")
        print(aov.anova_table)
        print("\n--- Task × Boundary interaction ---")
        print(aov.anova_table.loc[aov.anova_table.index.str.contains('Task:Boundary')])

    return aov

# run ANOVA on linear-only slice from the replica decoding (optional)
_ = run_task_boundary_roi_anova(acc_long_fig2,
                                out_csv=os.path.join(stats_output_path, 'anova_table_LINEAR_replica.csv'),
                                print_table=False)

#%% Figure 2A–C style plots (copied authors' style)

import numpy as np
import pandas as pd
import os
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.stats import sem

# paths
root_dir = Path(__file__).resolve().parents[1]
stats_output_path = os.path.join(root_dir, 'bold_decoding_anova_results')
fig_outdir = os.path.join(stats_output_path, "figs")
os.makedirs(fig_outdir, exist_ok=True)

# load your saved decoding CSV (already created by your pipeline)
acc_long_fig2 = pd.read_csv(os.path.join(stats_output_path, 'binary_withintask_ACC_long_FIG2_replica.csv'))

# plotting config
roi_order   = ['V1','V2','V3','V3AB','hV4','LO1','LO2','IPS']
task_order  = ['linear1','linear2','nonlinear']
task_labels = {'linear1':'Linear-1 Task','linear2':'Linear-2 Task','nonlinear':'Nonlinear Task'}
task_colors = {'linear1':'#1f4e79', 'linear2':'#2e86c1', 'nonlinear':'#76d7c4'}  # dark blue, blue, teal
chance_y    = 0.5

# small horizontal offsets so the three tasks don’t overlap
offsets = {'linear1': -0.18, 'linear2': 0.0, 'nonlinear': +0.18}

# panel defs to match A–C
panels = [
    ('linear1',  'A  Binary classifier: Predict "Linear-1" category'),
    ('linear2',  'B  Binary classifier: Predict "Linear-2" category'),
    ('nonlinear','C  Binary classifier: Predict "Nonlinear" category')
]

def plot_panel(boundary, title, show_legend=False):
    df_b = acc_long_fig2.query("Boundary == @boundary and Task in @task_order").copy()

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.axhline(chance_y, color='gray', linewidth=1.0)  # chance line

    x_base = np.arange(len(roi_order))

    # per-task plotting
    for task in task_order:
        df_t = df_b[df_b['Task'] == task]
        # subject-level gray dots
        for s, df_s in df_t.groupby('sub'):
            y = df_s.set_index('ROI').reindex(roi_order)['ACC'].values
            x = x_base + offsets[task]
            ax.scatter(x, y, s=12, color='#BFBFBF', zorder=1)  # light gray per-subject dots

        # means and sem
        g = df_t.groupby('ROI')['ACC']
        means = g.mean().reindex(roi_order).values
        errors = g.apply(lambda x: sem(x, nan_policy='omit')).reindex(roi_order).values
        x = x_base + offsets[task]

        # colored mean dot + error bar
        ax.errorbar(
            x, means, yerr=errors, fmt='o', markersize=6, capsize=3,
            linewidth=1.2, color=task_colors[task], ecolor=task_colors[task], label=task_labels[task], zorder=3
        )

    ax.set_xticks(x_base)
    ax.set_xticklabels(roi_order)
    ax.set_ylim(0.4, 1.0)
    ax.set_ylabel('Classifier accuracy')
    ax.set_title(title)
    if show_legend:
        leg = ax.legend(frameon=False, loc='upper right')
    fig.tight_layout()
    return fig, ax

# make the three panels; show legend on the last one (like your screenshot)
for i, (bnd, ttl) in enumerate(panels, 1):
    fig, ax = plot_panel(bnd, ttl, show_legend=(bnd == 'nonlinear'))
    out_png = os.path.join(fig_outdir, f"Figure2_{bnd}_replica_DOTSEM.png")
    fig.savefig(out_png, dpi=300, bbox_inches='tight')
    print(f"Saved: {out_png}")
    plt.show(fig)

