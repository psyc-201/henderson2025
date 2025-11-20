#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 17 22:04:40 2025

@author: lenakemmelmeier
"""

#%% Set up environment - load in appropriate packages

import numpy as np
import sys
import os
import gc
import torch
import time
import h5py
import torchvision.models as models
from sklearn import decomposition
import pandas as pd
import PIL.Image
import scipy.stats
import scipy.spatial.distance
import matplotlib.pyplot as plt
from pathlib import Path

from matplotlib import cm

# install from https://pypi.org/project/simclr/
# https://github.com/Spijkervet/SimCLR
import simclr
from simclr import SimCLR

# clip implemented in this package, from:
# https://github.com/openai/CLIP
import clip

#%% Set paths + init vars

root_dir = Path(__file__).resolve().parents[1]
image_sim_dir = os.path.join(root_dir, "image_similarity")
feat_dir = os.path.join(image_sim_dir, "features")

# only need this because it specifies where simclr pretrained weights are stored
simclr_weights_path = os.path.join(feat_dir,"simclr")

image_dir = os.path.join(root_dir, 'stimuli', 'new_blob_stim')

np.random.seed(0)


# pick device priority: MPS (Apple), then CUDA, then CPU
# this is for pytorch stuff
if torch.backends.mps.is_available():
    device = torch.device('mps')
    print('using device: mps')
elif torch.cuda.is_available():
    device = torch.device('cuda:0')
    print('using device: cuda:0')
else:
    device = torch.device('cpu')
    print('using device: cpu')

#%% Helper functions

def extract_features(image_data, \
                    block_inds, \
                    pooling_op = None,
                    save_dtype=np.float32,\
                    training_type='imgnet', \
                    batch_size=100, \
                    debug=False):
    """ 
    Extract features for resnet model.
    Image data is [n_images x 3 x n_pix x n_pix]
    Return features [n_images x n_channels]
    
    """

    debug = debug==1
    print('debug=%s'%debug)
    
    assert(len(block_inds)==1)
    # ll = block_inds[0]
    
    n_images = image_data.shape[0]
    
    model_architecture='RN50'
    
    do_tile=True
    model_filename=None
   
    n_batches = int(np.ceil(n_images/batch_size))

    # figure out how big features will be, by passing a test image through
    if do_tile:
        image_template = np.tile(image_data[0:1,:,:,:], [1,3,1,1])
    else:
        image_template = image_data[0:1,:,:,:]
    activ_template = get_resnet_activations_batch(image_template, block_inds, \
                                                 model_architecture, training_type, \
                                                  model_filename=model_filename, device=device)
    if pooling_op is not None:
        activ_template = [pooling_op(activ_template[0])]
    n_features_total = np.prod(activ_template[0].shape[1:])  
    print('number of features total: %d'%n_features_total)
    
    features = np.zeros((n_images, n_features_total),dtype=save_dtype)

    with torch.no_grad():

        
        for bb in range(n_batches):

            if debug and bb>1:
                continue
            print('Processing images for batch %d of %d'%(bb, n_batches))
            sys.stdout.flush()
            
            batch_inds = np.arange(batch_size * bb, np.min([batch_size * (bb+1), n_images]))

            # using grayscale images for better comparison w my other models.
            # need to tile to 3 so model weights will be right size
            if do_tile:
                image_batch = np.tile(image_data[batch_inds,:,:,:], [1,3,1,1])
            else:
                image_batch = image_data[batch_inds,:,:,:]

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
            activ_batch = get_resnet_activations_batch(image_batch, block_inds, \
                                                 model_architecture, training_type, \
                                                 model_filename=model_filename, device=device)
            if bb==0:
                print('size of activ this batch raw:')
                print(activ_batch[0].shape)
              
            if pooling_op is not None:
                
                activ_batch = [pooling_op(activ_batch[0])]
                
                if bb==0:
                    print('size of activ pooled:')
                    print(activ_batch[0].shape)
              
            activ_batch_reshaped = torch.reshape(activ_batch[0], [len(batch_inds), -1])
           
            if bb==0:
                print('size of activ reshaped:')
                print(activ_batch_reshaped.shape)
                
            features[batch_inds,:] = activ_batch_reshaped.detach().cpu().numpy()
        
    return features

def get_resnet_activations_batch(image_batch, \
                               block_inds, \
                               model_architecture, \
                               training_type, \
                               model_filename = None, \
                               device=None):

    """
    Get activations for images passed through pretrained resnet model.
    Specify which which layers to return.
    """

    if device is None:
        device = torch.device('cpu')
       
    if training_type=='clip':        
        
        print('Using CLIP model')
        model, preprocess = clip.load(model_architecture, device=device)
        model = model.visual
        
    elif training_type=='imgnet':
        
        # normal pre-trained model from pytorch
        print('Using pretrained Resnet50 model')
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2).float().to(device)
        
    elif training_type=='simclr':
        
        print('using SimCLR model')
        encoder = simclr.modules.get_resnet('resnet50', pretrained=False)
        projection_dim = 64
        n_features = encoder.fc.in_features  # get dimensions of last fully-connected layer
        model = SimCLR(encoder, projection_dim, n_features)
        model_fp = os.path.join(simclr_weights_path, 'checkpoint_100.tar')
        print('checkpoint path: %s' % model_fp)
        
        if device is None:
            device_use = torch.device('cpu')
        else:
            device_use = device
        
        print(device_use)
        model.load_state_dict(torch.load(model_fp, map_location=device_use))
        model = model.encoder.to(device_use)

    else:
        raise ValueError('training type %s not recognized' % training_type)
        
    model.eval()
    
    # The 16 residual blocks are segmented into 4 groups here, which have different numbers of features.
    blocks_each = [len(model.layer1), len(model.layer2), len(model.layer3), len(model.layer4)]
    which_group = np.repeat(np.arange(4), blocks_each)

    activ = [[] for ll in block_inds]
    hooks = [[] for ll in block_inds]
    
    # first making this subfunction that is needed to get the activation on a forward pass
    def get_activ_fwd_hook(ii, ll):
        def hook(self, input, output):
            # the relu operation is used multiple times per block, but we only 
            # want to save its output when it has this specific size.
            if output.shape[1] == n_features_each_resnet_block[ll]:
                print('executing hook for %s' % resnet_block_names[ll])  
                activ[ii] = output
                print(output.shape)
        return hook

    with torch.no_grad():

        # adding a "hook" to the module corresponding to each layer, so we'll save activations at each layer.
        for ii, ll in enumerate(block_inds):
            if which_group[ll] == 0:            
                h = model.layer1[ll].relu.register_forward_hook(get_activ_fwd_hook(ii, ll))
            elif which_group[ll] == 1:            
                h = model.layer2[ll - blocks_each[0]].relu.register_forward_hook(get_activ_fwd_hook(ii, ll))
            elif which_group[ll] == 2:            
                h = model.layer3[ll - sum(blocks_each[0:2])].relu.register_forward_hook(get_activ_fwd_hook(ii, ll))
            elif which_group[ll] == 3:            
                h = model.layer4[ll - sum(blocks_each[0:3])].relu.register_forward_hook(get_activ_fwd_hook(ii, ll))
            else:
                h = None
            hooks[ii] = h

        # >>> THIS IS THE MISSING PART: actually run the batch through the model <<<
        image_tensors = torch.from_numpy(image_batch).float().to(device)
        _ = model(image_tensors)

        # Now remove all the hooks
        for ii, ll in enumerate(block_inds):
            if hooks[ii] is not None:
                hooks[ii].remove()

    # Sanity check that we grabbed the right activations - check their sizes against expected
    exp_size = np.array(n_features_each_resnet_block)[block_inds]
    actual_size = [activ[bb].shape[1] for bb in range(len(activ))]
    assert(np.all(np.array(actual_size) == np.array(exp_size)))

    return activ


def get_resnet_features(debug=False, training_type='simclr', \
                        image_set_name = 'images'):
    
    # image_set_name = 'images_expt1'
    n_comp_keep = 500;
    

    if not os.path.exists(feat_dir):
        os.makedirs(feat_dir)


    if training_type=='imgnet':
        feat_path = os.path.join(feat_dir, 'resnet')
    elif training_type=='simclr':
        feat_path = os.path.join(feat_dir, 'simclr')
        
    if debug:
        feat_path = os.path.join(feat_path,'DEBUG')

    if not os.path.exists(feat_path):
        os.makedirs(feat_path)

    shape_df = make_shape_labels(image_dir)
    fn2save = os.path.join(feat_dir, 'Image_labels_grid3.csv')
    shape_df.to_csv(fn2save)

    image_data = load_images(shape_df, debug=debug)

    blocks_to_do = [2,6,12,15]
    
    # loop over resnet blocks
    for ll in blocks_to_do:

        block_inds = [ll]

        # if ll<=2:
        # reduce size of larger feature maps
        pooling_op = torch.nn.MaxPool2d(kernel_size=4, stride=4, padding=0)
        # else:
        #     pooling_op = None

        if debug:
            image_data_use = image_data[0:2,:,:,:]
        else:
            image_data_use = image_data
            
        # first extract features for all pixels/feature channels 
        features_raw = extract_features(image_data_use,\
                                        block_inds,\
                                        pooling_op = pooling_op, 
                                        save_dtype=np.float32,\
                                        training_type=training_type, \
                                        debug=debug)

        # reduce the dimensionality of the activs here
        scores, wts, pre_mean, ev = compute_pca(features_raw)

        n_keep = np.min([scores.shape[1], n_comp_keep])
        
        scores = scores[:,0:n_keep]

        feat_file_name = os.path.join(feat_path, \
                                      '%s_%s_block%d_pca.npy'%(image_set_name,\
                                                               training_type, \
                                                               ll))
        print('size of scores is:')
        print(scores.shape)
        print('saving to %s'%feat_file_name)
        np.save(feat_file_name, scores)

        
def compute_pca(values, max_pc_to_retain=None, copy_data=False):
    """
    Apply PCA to the data, return reduced dim data as well as weights, var explained.
    """
    n_features_actual = values.shape[1]
    n_trials = values.shape[0]
    
    if max_pc_to_retain is not None:        
        n_comp = np.min([np.min([max_pc_to_retain, n_features_actual]), n_trials])
    else:
        n_comp = np.min([n_features_actual, n_trials])
         
    print('Running PCA: original size of array is [%d x %d], dtype=%s'%\
          (n_trials, n_features_actual, values.dtype))
    sys.stdout.flush()
    t = time.time()
    pca = decomposition.PCA(n_components = n_comp, copy=copy_data)
    scores = pca.fit_transform(values)           
    elapsed = time.time() - t
    print('Time elapsed: %.5f'%elapsed)
    values = None            
    wts = pca.components_
    ev = pca.explained_variance_
    ev = ev/np.sum(ev)*100
    pre_mean = pca.mean_
  
    return scores, wts, pre_mean, ev

def prep_brick(debug=False):
    
    print('debug=%s'%debug)
    # make image brick that is used by get_gist.m
    
    shape_df = make_shape_labels(image_dir)
    image_data = load_images(shape_df, debug=debug)

    fn2save = os.path.join(feat_dir,'images_all.h5py')
    print('saving to %s'%fn2save)

    with h5py.File(fn2save, 'w') as f:

        f.create_dataset("stimuli", data=image_data, dtype='i')
        f.close()

def load_images(shape_df, debug=False):
    
    n_images = shape_df.shape[0]
    if debug:
        n_images_load = 10
    else:
        n_images_load = n_images
        
    first_image = PIL.Image.open(shape_df['filename_full'][0])
    n_pix = first_image.size[0]
    
    # [images x color_channels x height x width]
    image_array = np.zeros((n_images,1,n_pix,n_pix),dtype=np.float32)
    
    for ii in range(n_images_load):
        
        im = PIL.Image.open(shape_df['filename_full'][ii])
        imdat = np.reshape(np.array(im.getdata()), im.size)
        
        image_array[ii,0,:,:] = imdat
        
    return image_array

def make_shape_labels(image_dir):

    # create image labels based on Lena's 6x6 grid
    grid_ticks = np.array([0.10, 1.06, 2.02, 2.98, 3.94, 4.90])
    grid_x, grid_y = np.meshgrid(grid_ticks, grid_ticks)
    all_grid_points = np.column_stack((grid_x.ravel(), grid_y.ravel()))

    center = 2.5  # center of shape space is the "boundary"
    all_quadrant = np.zeros([np.shape(all_grid_points)[0], 1])
    all_quadrant[np.logical_and(all_grid_points[:, 0] > center,
                                all_grid_points[:, 1] > center)] = 1
    all_quadrant[np.logical_and(all_grid_points[:, 0] < center,
                                all_grid_points[:, 1] > center)] = 2
    all_quadrant[np.logical_and(all_grid_points[:, 0] < center,
                                all_grid_points[:, 1] < center)] = 3
    all_quadrant[np.logical_and(all_grid_points[:, 0] > center,
                                all_grid_points[:, 1] < center)] = 4

    labels_task1 = np.zeros(np.shape(all_quadrant))
    labels_task1[np.isin(all_quadrant, [1, 4])] = 2
    labels_task1[np.isin(all_quadrant, [2, 3])] = 1

    labels_task2 = np.zeros(np.shape(all_quadrant))
    labels_task2[np.isin(all_quadrant, [1, 2])] = 2
    labels_task2[np.isin(all_quadrant, [3, 4])] = 1

    labels_task3 = np.zeros(np.shape(all_quadrant))
    labels_task3[np.isin(all_quadrant, [1, 3])] = 2
    labels_task3[np.isin(all_quadrant, [2, 4])] = 1

    filenames_full = [
        os.path.join(image_dir, 'blob_%.2f_%.2f.png' % (x, y))
        for x, y in zip(all_grid_points[:, 0], all_grid_points[:, 1])
    ]

    shape_df = pd.DataFrame.from_dict({
        'coord_axis1': all_grid_points[:, 0],
        'coord_axis2': all_grid_points[:, 1],
        'quadrant': np.squeeze(all_quadrant),
        'labels_task1': np.squeeze(labels_task1),
        'labels_task2': np.squeeze(labels_task2),
        'labels_task3': np.squeeze(labels_task3),
        'filename_full': filenames_full
    })

    return shape_df

def get_main_grid():
    
    # start = 0;   
    # stop = 5;
    nsteps_main = 4;    
    start_grid = 0.1;
    stop_grid = 4.9;
    main_pts = np.round(np.linspace(start_grid,stop_grid, nsteps_main),1)
    [gridx,gridy] = np.meshgrid(main_pts,main_pts);
    main_grid_points = np.array([gridx.ravel(),gridy.ravel()]).T;

    return main_grid_points

def sep_and_sem(features, labels):
    labels = np.array(labels).astype(int)
    assert np.array_equal(np.unique(labels), np.array([1, 2]))
    g1 = features[labels == 1, :]
    g2 = features[labels == 2, :]

    within1 = scipy.spatial.distance.pdist(g1, metric='euclidean')
    within2 = scipy.spatial.distance.pdist(g2, metric='euclidean')
    within = np.concatenate([within1, within2], axis=0)
    between = scipy.spatial.distance.cdist(g1, g2, metric='euclidean').ravel()

    b = np.mean(between)
    w = np.mean(within)
    sep = (b - w) / (b + w)

    var_b = np.var(between, ddof=1)/between.size
    var_w = np.var(within, ddof=1)/within.size

    D = (b + w)
    df_db = (2.0 * w)/(D**2)
    df_dw = (-2.0 * b)/(D**2)
    var_sep = (df_db**2) * var_b + (df_dw**2) * var_w
    sem_sep = np.sqrt(var_sep)

    return sep, sem_sep
#%% step 1: make the image brick for matlab

prep_brick(debug=False)

# sanity check to make sure the brick is correct (has the image data)
fn = os.path.join(feat_dir, 'images_all.h5py')
print(fn)

with h5py.File(fn, 'r') as f:
    print('keys:', list(f.keys()))
    stim = f['stimuli'][:]  # load into memory

print('stimuli shape:', stim.shape)
print('dtype:', stim.dtype)
print('min, max:', np.min(stim), np.max(stim))

# show all images in a grid
n_images = stim.shape[0]
n_rows = int(np.ceil(np.sqrt(n_images)))
n_cols = int(np.ceil(n_images / n_rows))

plt.figure(figsize=(8, 8))
for i in range(n_images):
    ax = plt.subplot(n_rows, n_cols, i + 1)
    ax.imshow(stim[i, 0, :, :], cmap='gray')
    ax.axis('off')

plt.tight_layout()
plt.show()

#%% step 2: run the get_gist.m matlab script! and download simclr

# must run this before continuing to the next cell

# download the following file, place under the simclr dir: https://github.com/Spijkervet/SimCLR/releases/download/1.2/checkpoint_100.tar

#%% step 3: extract simclr features with resnet

# Define stuff about the resnet layers here
n_features_each_resnet_block = [256,256,256, 512,512,512,512, 1024,1024,1024,1024,1024,1024, 2048,2048,2048]
resnet_block_names = ['block%d'%nn for nn in range(len(n_features_each_resnet_block))]

get_resnet_features(debug=False, training_type='simclr', image_set_name='images')

simclr_feat_dir = os.path.join(feat_dir, "simclr")

# sanity check! did they save?
for bb in [2, 6, 12, 15]:
    fn = os.path.join(simclr_feat_dir, f"images_simclr_block{bb}_pca.npy")
    print(fn, "exists:", os.path.exists(fn))
    if os.path.exists(fn):
        arr = np.load(fn)
        print("  shape:", arr.shape)

#%% step 4: compute category separability + bar plot

# load gist features
fn = os.path.join(feat_dir, 'gist', 'images_gistdescriptors_4ori_4blocks.h5py')
print('loading GIST from:', fn)
with h5py.File(fn, 'r') as file:
    f = np.array(file['/features'])
f_gist = f
print('f_gist shape:', f_gist.shape)

# load simclr features (blocks 2, 6, 12, 15) and concatenate
blocks_to_do = [2, 6, 12, 15]
f = []
for bb in blocks_to_do:
    fn = os.path.join(feat_dir, 'simclr', 'images_simclr_block%d_pca.npy' % bb)
    print('loading SimCLR block %d from: %s' % (bb, fn))
    ftmp = np.load(fn)
    f.append(ftmp)
f = np.concatenate(f, axis=1)
f_simclr = f
print('f_simclr shape:', f_simclr.shape)

# load labels + get 4x4 main grid indices (for separability)
labs = pd.read_csv(os.path.join(feat_dir, 'Image_labels_grid3.csv'))
pts = np.array([labs['coord_axis1'], labs['coord_axis2']]).T

main_grid = get_main_grid()
gi = [np.argmin(np.sum((pts - g)**2, axis=1)) for g in main_grid]
gi = np.array(gi, dtype=int)
print('number of main grid points:', gi.size)

# subset to main grid for separability
f_gist_main = f_gist[gi, :]
f_simclr_main = f_simclr[gi, :]

# compute separability + sem for each task axis
gist_means = np.zeros((3,))
simclr_means = np.zeros((3,))
gist_sem = np.zeros((3,))
simclr_sem = np.zeros((3,))

for ai, axis in enumerate([1, 2, 3]):
    l = np.array(labs['labels_task%d' % axis])[gi].astype(int)
    m_gist, s_gist = sep_and_sem(f_gist_main, l)
    m_sim, s_sim = sep_and_sem(f_simclr_main, l)
    gist_means[ai] = m_gist
    simclr_means[ai] = m_sim
    gist_sem[ai] = s_gist
    simclr_sem[ai] = s_sim

print('gist_means:', gist_means)
print('simclr_means:', simclr_means)

# set up bar plot with 95% CI from sem
plt.rcParams['pdf.fonttype'] = 42

task_names = ['Linear (1)', 'Linear (2)', 'Nonlinear', 'Repeat']
task_colors = np.flipud(cm.GnBu(np.linspace(0, 1, 5))[1:, :])

vals = np.array([gist_means, simclr_means]).T  # (3 tasks x 2 models)
g_err = 1.96 * gist_sem
s_err = 1.96 * simclr_sem

xjitter = np.linspace(-0.25, 0.25, 3)
bw = np.diff(xjitter)[0] / 4 * 3

plt.figure(figsize=(5, 4))
ax = plt.subplot(1, 1, 1)

error_kw = dict(lw=0.8, capsize=3, capthick=0.8, ecolor='k')

for xi in range(2):
    yvals = vals[:, xi]
    if xi == 0:
        yerr = g_err
    else:
        yerr = s_err
    plt.bar(xi + np.array(xjitter), yvals, width=bw, color=task_colors[0:3, :], yerr=yerr, error_kw=error_kw)

plt.xticks(np.arange(2), ['GIST', 'SimCLR'])
plt.ylabel('Category separability (a.u.)', fontsize=12)
plt.yticks(fontsize=10)
plt.xticks(fontsize=10)

y_max = np.max(np.vstack([gist_means + g_err, simclr_means + s_err]))
plt.ylim([0, y_max * 1.2])

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# legend for task colors (linear1, linear2, nonlinear)
handles = [plt.Rectangle((0, 0), 1, 1, color=task_colors[i, :]) for i in range(3)]
ax.legend(handles, task_names[0:3], title='Task', frameon=False, loc='upper left', bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0)

figname = os.path.join(image_sim_dir, 'image_category_sep.png')
plt.savefig(figname, dpi=600, bbox_inches='tight')
print('saved:', figname)

source_df = pd.DataFrame(vals, index=task_names[0:3], columns=['GIST', 'SimCLR'])

#%% step 5: pca on gist features + scatter plots

# run pca on gist features (all 36 blobs)
scores, wts, pre_mean, ev = compute_pca(f_gist, max_pc_to_retain=100, copy_data=True)
print('scores shape:', scores.shape)

scores_all = scores

# pull axis coordinates from labels for coloring
axis1_vals = np.array(labs['coord_axis1'])
axis2_vals = np.array(labs['coord_axis2'])

vmin1, vmax1 = axis1_vals.min(), axis1_vals.max()
vmin2, vmax2 = axis2_vals.min(), axis2_vals.max()

cmap = cm.PuRd
norm1 = plt.Normalize(vmin=vmin1, vmax=vmax1)
norm2 = plt.Normalize(vmin=vmin2, vmax=vmax2)

# make 2-panel figure with separate horizontal colorbars
fig = plt.figure(figsize=(8, 4))
gs = fig.add_gridspec(2, 2, height_ratios=[4, 0.4], hspace=0.5, wspace=0.4)

ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[0, 1])
cax1 = fig.add_subplot(gs[1, 0])
cax2 = fig.add_subplot(gs[1, 1])

# colored by dimension 1
for ii in range(scores_all.shape[0]):
    ax1.plot(scores_all[ii, 0], scores_all[ii, 1], 'o', color=cmap(norm1(axis1_vals[ii])))
ax1.set_aspect('equal', 'box')
ax1.set_xticks([])
ax1.set_yticks([])
ax1.set_xlabel('PC 1')
ax1.set_ylabel('PC 2')
ax1.set_title('Colored by dimension 1')
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)

# colored by dimension 2
for ii in range(scores_all.shape[0]):
    ax2.plot(scores_all[ii, 0], scores_all[ii, 1], 'o', color=cmap(norm2(axis2_vals[ii])))
ax2.set_aspect('equal', 'box')
ax2.set_xticks([])
ax2.set_yticks([])
ax2.set_xlabel('PC 1')
ax2.set_ylabel('PC 2')
ax2.set_title('Colored by dimension 2')
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)

# colorbar for axis 1 coords
cb1 = plt.colorbar(cm.ScalarMappable(norm=norm1, cmap=cmap), cax=cax1, orientation='horizontal')
cb1.set_label('axis 1 coordinate (arb. units)', fontsize=10)
cb1.set_ticks([vmin1, vmax1])
cb1.set_ticklabels(['%.2f' % vmin1, '%.2f' % vmax1])
cax1.tick_params(labelsize=9)

# colorbar for axis 2 coords
cb2 = plt.colorbar(cm.ScalarMappable(norm=norm2, cmap=cmap), cax=cax2, orientation='horizontal')
cb2.set_label('axis 2 coordinate (arb. units)', fontsize=10)
cb2.set_ticks([vmin2, vmax2])
cb2.set_ticklabels(['%.2f' % vmin2, '%.2f' % vmax2])
cax2.tick_params(labelsize=9)

figname = os.path.join(image_sim_dir, 'image_gist_pca.png')
plt.savefig(figname, dpi=600, bbox_inches='tight')