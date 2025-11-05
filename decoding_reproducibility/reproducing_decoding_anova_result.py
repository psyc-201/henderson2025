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
from pathlib import Path # for infering path

# actual stats packages
from sklearn.linear_model import LogisticRegressionCV
from statsmodels.stats.anova import AnovaRM
from sklearn.feature_selection import f_classif

# for plotting
from scipy.stats import sem
# author plotting helpers (matches Fig 2 bar/dot style)

np.random.seed(0) # for reproducibility

#%% Set paths, import helpers from the authors'

# root_dir = '/Users/lenakemmelmeier/Documents/GitHub/henderson2025'

root_dir = Path(__file__).resolve().parents[1] # infer the root based on where this script is placed
os.chdir(root_dir)

os.chdir(root_dir)
package_dir = os.path.join(root_dir, 'decoding_reproducibility') # the parent folder containing code_utils (downloaded from authors)

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
data_utils.root = data_root

# plot utils lives next to data_util
from code_utils import plot_utils as _plot_utils
set_all_font_sizes = _plot_utils.set_all_font_sizes
plot_multi_bars = _plot_utils.plot_multi_bars

#%% Init other variables

num_participants = 10 # number of participants

# uses list comprehension to make a list of sub IDs to iterate over later
sub_ids = [str(i).zfill(2) for i in range(1, num_participants + 1)] # zero pads that sub IDs up to 2 digits (e.g., '1' becomes '01')

boundaries = {"linear1": [[1, 4], [2, 3]], "linear2": [[1, 2], [3, 4]], "nonlinear": [[1, 3], [2, 4]]} # dict for which quartiles correspond to which boundary groupings in the different conditions

#%% Helper functions (run this before the cells below!)

# call the authors' helper to load the main trial data
def load_main_data(sub_id):
    
    ss = int(sub_id)
    main_data, main_by_tr, main_labels, roi_names = data_utils.load_main_task_data(ss, make_time_resolved=False, use_bigIPS=True, concat_IPS=True)
    
    return main_data, main_labels, roi_names

# call the authors' helper to load the repeat trial data
def load_repeat_data(sub_id):
    ss = int(sub_id)
    
    rep_data, rep_by_tr, rep_labels, roi_names = data_utils.load_repeat_task_data(
        ss, make_time_resolved=False, use_bigIPS=True, concat_IPS=True
    )
    
    return rep_data, rep_labels, roi_names

# rename IPSall to IPS so plotting code is same as authors'
def rename_ips(roi_names):
    return ["IPS" if r == "IPSall" else r for r in roi_names]

# makes y for a given boundary (maps quadrant -> 0/1)
def make_binary_labels(q_arr, boundary_name, boundaries):

    g0, g1 = boundaries[boundary_name]
    y = np.full(q_arr.shape, -1, dtype=int)
    y[np.isin(q_arr, g0)] = 0
    y[np.isin(q_arr, g1)] = 1
    
    return y

# leave-one-run-out decoding within task, per ROI
def decode_within_task_one_roi(X, y, runs, is_main_mask):
    
    clf = LogisticRegressionCV(Cs=10, cv=3, penalty='l2', solver='liblinear', max_iter=200, n_jobs=1, class_weight='balanced')
    accs = []
    for r in np.unique(runs):
        tr = (runs != r) & is_main_mask
        te = (runs == r) & is_main_mask  # score on main-grid only
        if tr.sum() < 4 or np.unique(y[tr]).size < 2 or te.sum() == 0:
            continue
        clf.fit(X[tr], y[tr])
        accs.append((clf.predict(X[te]) == y[te]).mean())
    return float(np.mean(accs)) if accs else np.nan

def subject_decoding_df(sub_id, boundaries, top_k_voxels=None, mode='fig2'):

    # load main and repeat data
    main_rois, main_label, main_roi_names = load_main_data(sub_id)
    rep_rois,  rep_label,  rep_roi_names  = load_repeat_data(sub_id)

    # just renaming the ips roi
    roi_names = rename_ips(main_roi_names)

    # get label vectors
    is_main_grid_main = main_label['is_main_grid'].astype(bool).to_numpy()
    runs_main = main_label['run_overall'].to_numpy().astype(int)
    tasks_main = main_label['task'].to_numpy().astype(int)
    quads_main = main_label['quadrant'].to_numpy().astype(int)

    # get repeat labels (voxel selection)
    is_main_grid_rep = rep_label['is_main_grid'].astype(bool).to_numpy()
    quads_rep = rep_label['quadrant'].to_numpy().astype(int)

    voxel_masks = []
    for X_rep_roi in rep_rois:
        
        # X_rep_roi: [nTrials x nVox] (already run-zscored & trial-averaged by authors)
        Xr = X_rep_roi[is_main_grid_rep, :]
        qr = quads_rep[is_main_grid_rep]

        if top_k_voxels in (None, 'all'):
            mask = np.ones(Xr.shape[1], dtype=bool)
        else:
            nvox = Xr.shape[1]
            top_k = int(min(nvox, np.min([r.shape[1] for r in rep_rois])))
            if isinstance(top_k_voxels, (int, np.integer)):
                top_k = min(top_k, int(top_k_voxels))
            mask = select_voxels_by_quadrant_anova(Xr, qr, top_k)
        voxel_masks.append(mask)

    # apply the voxel masks
    roi_arrays = []
    roi_names_kept = []
    
    for rname, X_main_roi, m in zip(roi_names, main_rois, voxel_masks):
        
        if m.sum() == 0:
            continue
        
        roi_arrays.append(X_main_roi[:, m])   # still [nTrials x nKeptVox]
        roi_names_kept.append(rname)
        
    roi_names = roi_names_kept

    # make the trial subsets
    def near_far_masks(labels_df):
        # reuse your helper to compute near/far; expects the CSV columns present
        return build_near_far_masks(labels_df)

    if mode == 'anova_near':
        nf = near_far_masks(main_label)  # dict by task_id
    rows = []
    task_id_to_name = {1:'linear1', 2:'linear2', 3:'nonlinear'}

    tasks_to_run = (1,2) if mode == 'anova_near' else (1,2,3)
    boundaries_to_try = ('linear1','linear2') if mode != 'fig2' else ('linear1','linear2','nonlinear')

    for task_id in tasks_to_run:
        if mode == 'fig2':
            tmask = (tasks_main == task_id) & is_main_grid_main
        else:
            tmask = nf[task_id]['near']  # NEAR-only

        if tmask.sum() == 0:
            continue

        q_t = quads_main[tmask]
        run_t = runs_main[tmask]

        for bname in boundaries_to_try if mode == 'fig2' else ('linear1','linear2'):
            y = make_binary_labels(q_t, bname, boundaries)
            valid = (y >= 0)
            yv, rv = y[valid], run_t[valid]

            # simple all-True mask because we've already subset trials
            is_main_mask = np.ones_like(yv, dtype=bool)

            for rname, X_roi in zip(roi_names, roi_arrays):
                Xv = X_roi[tmask, :][valid]
                acc = decode_within_task_one_roi(Xv, yv, rv, is_main_mask=is_main_mask)
                rows.append([sub_id, rname, task_id_to_name[task_id], bname, acc])

    print(f"[decode] Finished decoding for subject {sub_id}")
    return pd.DataFrame(rows, columns=['sub','ROI','Task','Boundary','ACC'])

# three-way anova w/ ROI x Task x Boundary (only for linear tasks/boundaries)
def run_task_boundary_roi_anova(df_long, out_csv, print_table = False):

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

# one-way anova for voxel selection
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

# masks for trials -- which are near vs. far for the different boundary conditions
def build_near_far_masks(labels):
    
    L = labels.copy()
    mg = L['is_main_grid'].astype(bool).to_numpy()
    t = L['task'].astype(int).to_numpy()
    x = pd.to_numeric(L['nn_ptx'], errors='coerce').to_numpy()
    y = pd.to_numeric(L['nn_pty'], errors='coerce').to_numpy()
    d1 = pd.to_numeric(L['dist_from_bound1'], errors='coerce').to_numpy()
    d2 = pd.to_numeric(L['dist_from_bound2'], errors='coerce').to_numpy()
    keys = np.array(list(zip(np.round(x,3), np.round(y,3))), dtype=object)

    masks = {1:{'near':np.zeros(len(L),bool),'far':np.zeros(len(L),bool)},
             2:{'near':np.zeros(len(L),bool),'far':np.zeros(len(L),bool)},
             3:{'near':np.zeros(len(L),bool),'far':np.zeros(len(L),bool)}}

    for task_id in (1,2):
        
        use = (t==task_id) & mg & np.isfinite(x) & np.isfinite(y)
        
        if not use.any(): continue
        d = d1 if task_id==1 else d2
        
        df = pd.DataFrame({'k':keys[use], 'dist':d[use]})
        pos = df.groupby('k', as_index=False)['dist'].median().sort_values('dist', kind='mergesort')
        nearK = pos['k'].iloc[:8].tolist()
        farK  = pos['k'].iloc[-8:].tolist()
        
        near_mask = np.array([k in set(nearK) for k in keys], dtype=bool)
        far_mask  = np.array([k in set(farK)  for k in keys], dtype=bool)
        
        masks[task_id]['near'] = use & near_mask
        masks[task_id]['far']  = use & far_mask


    # nonlinear: corners far
    use = (t==3) & mg & np.isfinite(x) & np.isfinite(y)
    if use.any():
        
        xs = np.unique(np.round(x[use],3)); ys = np.unique(np.round(y[use],3))
        
        if xs.size>=2 and ys.size>=2:
            corners = {(xs.min(),ys.min()),(xs.min(),ys.max()),(xs.max(),ys.min()),(xs.max(),ys.max())}
            is_corner = np.array([k in corners for k in keys], dtype=bool)
            masks[3]['far']  = use & is_corner
            masks[3]['near'] = use & (~is_corner)
            
    return masks

#%% Decode each subject for Fig 2 (no near/far split)

acc_rows_fig2 = [subject_decoding_df(s, boundaries, top_k_voxels=None, mode='fig2') for s in sub_ids]
acc_long_fig2 = pd.concat(acc_rows_fig2, ignore_index=True)
acc_long_fig2.to_csv(os.path.join(stats_output_path, 'binary_within_task_acc_long_FIG2.csv'), index=False)

#%% Decode each subject for ANOVA (near only: labels are linear1 vs. linear2)

acc_rows_near = [subject_decoding_df(s, boundaries, top_k_voxels=None, mode='anova_near') for s in sub_ids]
acc_long_near = pd.concat(acc_rows_near, ignore_index=True)
acc_long_near.to_csv(os.path.join(stats_output_path, 'binary_within_task_acc_long_NEARONLY.csv'), index=False)

#%% Run the 3-way ANOVA (ROI × Task × Boundary) on the data from above

aov = run_task_boundary_roi_anova(acc_long_near, out_csv=os.path.join(stats_output_path, 'anova_table_NEARONLY.csv'))

#%% Figure 2A–C style plots (I copied the authors' style here)

fig_outdir = os.path.join(stats_output_path, "figs")
os.makedirs(fig_outdir, exist_ok=True)
set_all_font_sizes(12)

# panels in order
panels = [('linear1', 'A  Binary classifier: Predict "Linear-1" category'), ('linear2', 'B  Binary classifier: Predict "Linear-2" category'),('nonlinear', 'C  Binary classifier: Predict "Nonlinear" category')]

task_order = ['linear1','linear2','nonlinear']
task_labels = ['Linear-1 Task','Linear-2 Task','Nonlinear Task']
roi_order = ['V1','V2','V3','V3AB','hV4','LO1','LO2','IPS']

for bnd, title in panels:
    
    df_b = acc_long_fig2[acc_long_fig2['Boundary']==bnd].copy()
    df_b = df_b[df_b['Task'].isin(task_order)]
    
    # get mean/sem arrays [nROI x nTask]
    rois_here = [r for r in roi_order if r in df_b['ROI'].unique()]
    means = np.zeros((len(rois_here), len(task_order)))
    errs  = np.zeros((len(rois_here), len(task_order)))
    
    # subject-level points organized as [nSub x nROI x nTask]
    subs = sorted(df_b['sub'].unique().tolist())
    pts  = np.full((len(subs), len(rois_here), len(task_order)), np.nan)

    for j, t in enumerate(task_order):
        
        grp = df_b[df_b['Task']==t]
        g = grp.groupby('ROI')['ACC']
        m = g.mean().reindex(rois_here)
        e = g.apply(lambda x: sem(x, nan_policy='omit')).reindex(rois_here)
        means[:, j] = m.values
        errs[:,  j] = e.values
        
        # fill per-subject points
        for si, s in enumerate(subs):
            gv = grp[grp['sub']==s].set_index('ROI')['ACC'].reindex(rois_here)
            pts[si, :, j] = gv.values

    fh = plot_multi_bars(means, err_data=errs, point_data=pts, add_ss_lines=True, xticklabels=rois_here, ylabel='Classifier accuracy', ylim=[0.4, 1.0], horizontal_line_pos=0.5, title=title, legend_labels=task_labels, legend_overlaid=False, legend_separate=False, err_capsize=3, fig_size=(10,4))
    
    out_png = os.path.join(fig_outdir, f"Figure2_{bnd}.png")
    fh.savefig(out_png, dpi=300, bbox_inches='tight')
    print(f"Saved: {out_png}")
