#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  3 14:43:25 2025

@author: lenakemmelmeier
"""

#%% Set up environment - load in appropriate packages

import numpy as np
import os
import sys
import pandas as pd
from pathlib import Path
import math

from sklearn.model_selection import LeaveOneGroupOut
from sklearn.linear_model import LogisticRegressionCV
from statsmodels.stats.anova import AnovaRM

from scipy.stats import sem
import matplotlib.pyplot as plt

import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)

np.random.seed(0)

root_dir = Path(__file__).resolve().parents[1]
stats_output_path = os.path.join(root_dir, 'bold_decoding_anova_results')
fig_outdir = os.path.join(stats_output_path, "figs")
os.makedirs(fig_outdir, exist_ok=True)

#%% Set paths, import helpers from the authors

root_dir = Path(__file__).resolve().parents[1]
os.chdir(root_dir)

package_dir = os.path.join(root_dir, 'decoding_reproducibility')

stats_output_path = os.path.join(root_dir, 'bold_decoding_anova_results')
os.makedirs(stats_output_path, exist_ok=True)

if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

if os.path.isdir(os.path.join(package_dir, 'code_utils')):
    if package_dir not in sys.path:
        sys.path.insert(0, package_dir)

from code_utils import data_utils
data_root = os.path.join(root_dir, 'data')
data_utils.root = data_root

#%% Init variables

num_participants = 10
sub_ids = [str(i).zfill(2) for i in range(1, num_participants + 1)]

boundaries = {
    "linear1": [[1, 4], [2, 3]],
    "linear2": [[1, 2], [3, 4]],
    "nonlinear": [[1, 3], [2, 4]],
}

task_id_to_name = {1: 'linear1', 2: 'linear2', 3: 'nonlinear', 4: 'repeat'}

#%% Helper functions

def load_main_data(sub_id):
    ss = int(sub_id)
    main_data, main_by_tr, main_labels, roi_names = data_utils.load_main_task_data(
        ss, make_time_resolved=False, use_bigIPS=True, concat_IPS=True
    )
    return main_data, main_labels, roi_names

def load_repeat_data(sub_id):
    ss = int(sub_id)
    rep_data, rep_by_tr, rep_labels, roi_names = data_utils.load_repeat_task_data(
        ss, make_time_resolved=False, use_bigIPS=True, concat_IPS=True
    )
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

def decode_within_task_one_roi_replica(data_concat, quad_labs, task_labs, cv_labs, is_main_grid, task_id, boundary_name):
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

def decode_with_subset(data_concat, quad_labs, task_labs, cv_labs, is_main_grid, task_id, boundary_name, subset_mask):
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

#%% subject-wise decoding (near/far + full grid)

def subject_decoding_df_replica(sub_id):
    print(f"[decode] Started decoding for subject {sub_id}")

    main_rois, main_lab, main_roi_names = load_main_data(sub_id)
    rep_rois, rep_lab, rep_roi_names = load_repeat_data(sub_id)

    roi_names = rename_ips(main_roi_names)
    n_rois = len(roi_names)

    concat_labels = pd.concat([main_lab, rep_lab], axis=0)

    cv_main = main_lab['run_overall'].to_numpy().astype(int)
    cv_rep = rep_lab['run_overall'].to_numpy().astype(int) + cv_main.max()
    cv_labs = np.concatenate([cv_main, cv_rep], axis=0)

    is_main_grid = (concat_labels['is_main_grid'].to_numpy().astype(int) == 1)
    quad_labs = concat_labels['quadrant'].to_numpy().astype(int)
    task_labs = concat_labels['task'].to_numpy().astype(int)
    task_labs[~np.isin(task_labs, [1,2,3])] = 4

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
    f = math.sqrt(eta_p2 / (1.0 - eta_p2))  # cast as a float (Cohen's f)
    return eta_p2, f

# compute effect sizes row-wise and add to table
eta_vals = []
f_vals = []

for Fi, d1, d2 in zip(anova_result['F Value'], anova_result['Num DF'], anova_result['Den DF']):
    eta_p2, f = effect_sizes_from_F(Fi, d1, d2)
    eta_vals.append(eta_p2)
    f_vals.append(f)

anova_result['eta_p2'] = eta_vals
anova_result['cohens_f'] = f_vals

# csv w/ effect sizes
anova_result.to_csv(os.path.join(
    stats_output_path, 'anova_table_LINEAR_NEARONLY_replica_with_effectsizes.csv'
), index=False)

print(anova_result)

#%% apa-style anova table (image; no cohens f)

def save_apa_anova_table(anova_result, out_png):

    df = anova_result.copy()

    # get effect names
    if 'Unnamed: 0' in df.columns:
        df = df.rename(columns={'Unnamed: 0': 'Effect'})
    else:
        df = df.reset_index().rename(columns={'index': 'Effect'})

    # format
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
        'eta_p2': 'np²'   # change label here if you want ηp² instead
    })
    df = df[['Effect','df1','df2','F','p','np²']]

    # figure
    n_rows = len(df)
    fig, ax = plt.subplots(figsize=(6, 0.6 + 0.3 * n_rows))
    ax.axis('off')

    # table with no gridlines (we'll draw our own rules)
    table = ax.table(
        cellText=df.values,
        colLabels=df.columns,
        loc='center',
        cellLoc='right'  # start with right align, fix Effect below
    )

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    for col in range(len(df.columns)):
        table.auto_set_column_width(col)

    # remove all cell borders, set alignments + header style
    n_cols = len(df.columns)
    for (r, c), cell in table.get_celld().items():
        cell.visible_edges = ''          # no borders from table
        cell.set_height(0.12)
        if r == 0:
            cell.set_text_props(weight='bold')
            cell.set_facecolor('#F2F2F2')
        else:
            cell.set_facecolor('white')

        # effect column left-aligned
        if c == 0:
            cell._loc = 'left'
            cell._text.set_ha('left')
        else:
            cell._loc = 'right'
            cell._text.set_ha('right')

    # coordinates for rules
    left = table[(0, 0)].get_x()
    right = table[(0, n_cols - 1)].get_x() + table[(0, n_cols - 1)].get_width()

    top_header = table[(0, 0)].get_y() + table[(0, 0)].get_height()
    bottom_header = table[(0, 0)].get_y()
    bottom_table = table[(n_rows, 0)].get_y()

    # draw apa-style horizontal lines
    ax.hlines(top_header,   left, right, linewidth=1.5, color='black')  # top
    ax.hlines(bottom_header,left, right, linewidth=1.0, color='black')  # under header
    ax.hlines(bottom_table, left, right, linewidth=1.5, color='black')  # bottom

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
task_labels = {'linear1':'Linear-1 Task','linear2':'Linear-2 Task','nonlinear':'Nonlinear Task'}
task_colors = {'linear1':'#1f4e79','linear2':'#2e86c1','nonlinear':'#76d7c4'}
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