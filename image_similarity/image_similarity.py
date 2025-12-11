#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 17 22:04:40 2025

@author: lenakemmelmeier

Extension: inspecting the dimensionality of the new shape stimuli and their category separability
"""

#%% set up environment - load in appropriate packages

import numpy as np
import sys # basic system utilities (e.g., stdout, exit)
import os
import gc # manual garbage collection to keep memory clean between batches
import torch # core deep learning library (resnet, simclr, device handling)
import time # simple timing of pca / feature extraction
import h5py # read/write hdf5 files (image brick, gist features)
import torchvision.models as models # pretrained resnet backbones
from sklearn import decomposition # pca implementation
import pandas as pd
import PIL.Image # loading png stimuli from disk
import scipy.stats # general stats helpers (not heavily used here)
import scipy.spatial.distance # distance metrics for category separability
import matplotlib.pyplot as plt
from pathlib import Path # cleaner path handling
from matplotlib import cm # colormaps for plots

# install from https://pypi.org/project/simclr/
# https://github.com/Spijkervet/SimCLR
import simclr # simclr training utilities and resnet encoders
from simclr import SimCLR # simclr model wrapper around the encoder

# clip implemented in this package, from:
# https://github.com/openai/CLIP
import clip # alternative visual encoder option (not the main path here)


#%% set paths + init vars

# figure out project root from this script location
root_dir = Path(__file__).resolve().parents[1]

# dir where image-similarity outputs will live
image_sim_dir = os.path.join(root_dir, "image_similarity")

# dir where all feature files (gist, simclr, etc.) will be stored
feat_dir = os.path.join(image_sim_dir, "features")

# where simclr pretrained weights are stored
simclr_weights_path = os.path.join(feat_dir, "simclr")

# dir with new stim
image_dir = os.path.join(root_dir, "stimuli", "new_blob_stim")

# set a global seed so things are reproducible/deterministic
np.random.seed(0)

# pick device priority for torch: mps (apple), then cuda, then cpu
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("using device: mps")
elif torch.cuda.is_available():
    device = torch.device("cuda:0")
    print("using device: cuda:0")
else:
    device = torch.device("cpu")
    print("using device: cpu")

#%% helper functions (feature extraction, pca, image brick, labels, separability)

# note: resnet/simclr feature-extraction code below is adapted very very closely from
# henderson et al. (2025) so that behavior matches their original pipeline! 

def extract_features(image_data,
                     block_inds,
                     pooling_op=None,
                     save_dtype=np.float32,
                     training_type="imgnet",
                     batch_size=100,
                     debug=False):
    """
    extract features from a resnet-like model for a batch of images.
    """

    debug = debug == 1
    print("debug=%s" % debug)

    # only support one block index at a time (mirrors original code)
    assert len(block_inds) == 1

    n_images = image_data.shape[0]

    # architecture string used by helper that builds the model
    model_architecture = "RN50"

    # images are grayscale so we tile to 3 channels to satisfy resnet weights
    do_tile = True

    # path for custom model weights (not used here, but kept for compatibility)
    model_filename = None

    # how many batches we need to cover all images
    n_batches = int(np.ceil(n_images / batch_size))

    # figure out feature dimensionality by running a single test image
    if do_tile:
        image_template = np.tile(image_data[0:1, :, :, :], [1, 3, 1, 1])
    else:
        image_template = image_data[0:1, :, :, :]

    activ_template = get_resnet_activations_batch(
        image_template,
        block_inds,
        model_architecture,
        training_type,
        model_filename=model_filename,
        device=device
    )

    # spatial pooling can shrink feature maps before flattening
    if pooling_op is not None:
        activ_template = [pooling_op(activ_template[0])]

    # flatten everything except batch dimension to get total feature count
    n_features_total = np.prod(activ_template[0].shape[1:])
    print("number of features total: %d" % n_features_total)

    # preallocate feature array [n_images x n_features]
    features = np.zeros((n_images, n_features_total), dtype=save_dtype)

    # no gradients needed, we are only doing forward passes
    with torch.no_grad():

        for bb in range(n_batches):

            if debug and bb > 1:
                continue

            print("processing images for batch %d of %d" % (bb, n_batches))
            sys.stdout.flush()

            # indices of images in this batch
            batch_inds = np.arange(
                batch_size * bb,
                np.min([batch_size * (bb + 1), n_images])
            )

            # tile grayscale images into 3 channels to match resnet input
            if do_tile:
                image_batch = np.tile(image_data[batch_inds, :, :, :], [1, 3, 1, 1])
            else:
                image_batch = image_data[batch_inds, :, :, :]

            # clean up gpu memory between batches
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # run images through resnet and grab activations at requested block
            activ_batch = get_resnet_activations_batch(
                image_batch,
                block_inds,
                model_architecture,
                training_type,
                model_filename=model_filename,
                device=device
            )

            if bb == 0:
                print("size of activ this batch raw:")
                print(activ_batch[0].shape)

            # spatial pooling step if a pooling op is given
            if pooling_op is not None:
                activ_batch = [pooling_op(activ_batch[0])]

                if bb == 0:
                    print("size of activ pooled:")
                    print(activ_batch[0].shape)

            # flatten feature map for this batch into [batch_size x n_features]
            activ_batch_reshaped = torch.reshape(activ_batch[0], [len(batch_inds), -1])

            if bb == 0:
                print("size of activ reshaped:")
                print(activ_batch_reshaped.shape)

            # copy from torch to numpy for storage
            features[batch_inds, :] = activ_batch_reshaped.detach().cpu().numpy()

    return features


def get_resnet_activations_batch(image_batch,
                                 block_inds,
                                 model_architecture,
                                 training_type,
                                 model_filename=None,
                                 device=None):
    """
    get activations for images passed through a pretrained resnet-like model.
    we use forward hooks to collect activations from specific residual blocks.
    """

    if device is None:
        device = torch.device("cpu")

    # choose which type of pretrained model we are using
    if training_type == "clip":
        print("using clip model")
        model, preprocess = clip.load(model_architecture, device=device)
        model = model.visual

    elif training_type == "imgnet":
        # standard resnet50 pretrained on imagenet
        print("using pretrained resnet50 model")
        model = models.resnet50(
            weights=models.ResNet50_Weights.IMAGENET1K_V2
        ).float().to(device)

    elif training_type == "simclr":
        print("using simclr model")

        # simclr encoder is a resnet backbone with projection head
        encoder = simclr.modules.get_resnet("resnet50", pretrained=False)
        projection_dim = 64 # dimension of projection head (not used directly here)
        n_features = encoder.fc.in_features # dimension of last fully-connected layer

        model = SimCLR(encoder, projection_dim, n_features)

        model_fp = os.path.join(simclr_weights_path, "checkpoint_100.tar")
        print("checkpoint path: %s" % model_fp)

        if device is None:
            device_use = torch.device("cpu")
        else:
            device_use = device

        print(device_use)
        model.load_state_dict(torch.load(model_fp, map_location=device_use))
        model = model.encoder.to(device_use)

    else:
        raise ValueError("training type %s not recognized" % training_type)

    model.eval()

    # the residual blocks are segmented into 4 groups, each with a different feature count
    blocks_each = [len(model.layer1), len(model.layer2), len(model.layer3), len(model.layer4)]
    which_group = np.repeat(np.arange(4), blocks_each)

    # these globals must already exist (defined later, before calling this)
    # n_features_each_resnet_block and resnet_block_names describe each block
    activ = [[] for _ in block_inds] # will hold activations for each requested block
    hooks = [[] for _ in block_inds] # handles so we can remove hooks later

    # nested function to build a forward hook for each block
    def get_activ_fwd_hook(ii, ll):
        def hook(self, input, output):
            # relu is used multiple times per block, but we only want output
            # when the feature dimension matches the known size for this block
            if output.shape[1] == n_features_each_resnet_block[ll]:
                print("executing hook for %s" % resnet_block_names[ll])
                activ[ii] = output
                print(output.shape)
        return hook

    with torch.no_grad():

        # attach hooks to the right relu in each residual block
        for ii, ll in enumerate(block_inds):

            if which_group[ll] == 0:
                h = model.layer1[ll].relu.register_forward_hook(get_activ_fwd_hook(ii, ll))

            elif which_group[ll] == 1:
                h = model.layer2[ll - blocks_each[0]].relu.register_forward_hook(
                    get_activ_fwd_hook(ii, ll)
                )

            elif which_group[ll] == 2:
                h = model.layer3[ll - sum(blocks_each[0:2])].relu.register_forward_hook(
                    get_activ_fwd_hook(ii, ll)
                )

            elif which_group[ll] == 3:
                h = model.layer4[ll - sum(blocks_each[0:3])].relu.register_forward_hook(
                    get_activ_fwd_hook(ii, ll)
                )

            else:
                h = None

            hooks[ii] = h

        # put numpy batch onto device and run forward pass to trigger hooks
        image_tensors = torch.from_numpy(image_batch).float().to(device)
        _ = model(image_tensors)

        # remove hooks so we do not keep accumulating them
        for ii, ll in enumerate(block_inds):
            if hooks[ii] is not None:
                hooks[ii].remove()

    # sanity check that we grabbed the expected feature sizes
    exp_size = np.array(n_features_each_resnet_block)[block_inds]
    actual_size = [activ_block.shape[1] for activ_block in activ]
    assert np.all(np.array(actual_size) == np.array(exp_size))

    return activ


def get_resnet_features(debug=False,
                        training_type="simclr",
                        image_set_name="images"):
    """
    wrapper to extract resnet/simclr features for all images and do pca.
    """

    n_comp_keep = 500 # number of principal components to keep per block

    # make sure the feature directory exists
    if not os.path.exists(feat_dir):
        os.makedirs(feat_dir)

    # pick output subdirectory based on training type
    if training_type == "imgnet":
        feat_path = os.path.join(feat_dir, "resnet")
    elif training_type == "simclr":
        feat_path = os.path.join(feat_dir, "simclr")
    else:
        raise ValueError("training_type %s not supported in get_resnet_features" % training_type)

    # debug-specific subfolder
    if debug:
        feat_path = os.path.join(feat_path, "DEBUG")

    if not os.path.exists(feat_path):
        os.makedirs(feat_path)

    # create dataframe with coordinates, quadrants, labels, and filenames
    shape_df = make_shape_labels(image_dir)

    # save labels so they can be reused later
    fn2save = os.path.join(feat_dir, "Image_labels_grid3.csv")
    shape_df.to_csv(fn2save)

    # load image data as [n_images x 1 x height x width]
    image_data = load_images(shape_df, debug=debug)

    # resnet blocks we will extract features from (matching henderson et al.)
    blocks_to_do = [2, 6, 12, 15]

    # loop over resnet blocks and extract features + pca scores
    for ll in blocks_to_do:

        block_inds = [ll]

        # reduce spatial size of larger feature maps with pooling
        pooling_op = torch.nn.MaxPool2d(kernel_size=4, stride=4, padding=0)

        # debug mode only uses first few images
        if debug:
            image_data_use = image_data[0:2, :, :, :]
        else:
            image_data_use = image_data

        # first extract raw features for this block
        features_raw = extract_features(
            image_data_use,
            block_inds,
            pooling_op=pooling_op,
            save_dtype=np.float32,
            training_type=training_type,
            debug=debug
        )

        # run pca to reduce dimensionality
        scores, wts, pre_mean, ev = compute_pca(features_raw)

        # keep at most n_comp_keep pcs
        n_keep = np.min([scores.shape[1], n_comp_keep])
        scores = scores[:, 0:n_keep]

        # save pc scores for this block
        feat_file_name = os.path.join(
            feat_path,
            "%s_%s_block%d_pca.npy" % (image_set_name, training_type, ll)
        )
        print("size of scores is:")
        print(scores.shape)
        print("saving to %s" % feat_file_name)
        np.save(feat_file_name, scores)


def compute_pca(values,
                max_pc_to_retain=None,
                copy_data=False):
    """
    apply pca to an array and return scores, weights, mean, and var explained.
    """

    n_features_actual = values.shape[1]
    n_trials = values.shape[0]

    # figure out how many pcs we can and should keep
    if max_pc_to_retain is not None:
        n_comp = np.min([np.min([max_pc_to_retain, n_features_actual]), n_trials])
    else:
        n_comp = np.min([n_features_actual, n_trials])

    print(
        "running pca: original size of array is [%d x %d], dtype=%s"
        % (n_trials, n_features_actual, values.dtype)
    )
    sys.stdout.flush()

    # time the pca so we know how long it takes
    t = time.time()
    pca = decomposition.PCA(n_components=n_comp, copy=copy_data)

    # scores: trial-by-pc representation
    scores = pca.fit_transform(values)

    elapsed = time.time() - t
    print("time elapsed: %.5f" % elapsed)

    # free original data if desired
    values = None

    # pca components (feature weights) and explained variance
    wts = pca.components_
    ev = pca.explained_variance_
    ev = ev / np.sum(ev) * 100 # percent variance explained per pc
    pre_mean = pca.mean_

    return scores, wts, pre_mean, ev


def prep_brick(debug=False):
    """
    build an image brick (hdf5 file) for use by the matlab gist code.
    """

    print("debug=%s" % debug)

    # make labels and filenames for the full grid
    shape_df = make_shape_labels(image_dir)

    # load corresponding images into memory
    image_data = load_images(shape_df, debug=debug)

    # this file is what the matlab get_gist.m script expects
    fn2save = os.path.join(feat_dir, "images_all.h5py")
    print("saving to %s" % fn2save)

    with h5py.File(fn2save, "w") as f:
        f.create_dataset("stimuli", data=image_data, dtype="i")
        f.close()


def load_images(shape_df,
                debug=False):
    """
    load lena's blob images into a 4d numpy array [n_images x 1 x height x width].
    """

    n_images = shape_df.shape[0]

    # in debug mode, only load a small subset
    if debug:
        n_images_load = 10
    else:
        n_images_load = n_images

    # open the first image to get pixel dimensions
    first_image = PIL.Image.open(shape_df["filename_full"][0])
    n_pix = first_image.size[0]

    # allocate array for grayscale image data
    image_array = np.zeros((n_images, 1, n_pix, n_pix), dtype=np.float32)

    # loop over images and load pixel values
    for ii in range(n_images_load):

        im = PIL.Image.open(shape_df["filename_full"][ii])
        imdat = np.reshape(np.array(im.getdata()), im.size)

        image_array[ii, 0, :, :] = imdat

    return image_array


def make_shape_labels(image_dir):
    """
    build a dataframe with coordinates, quadrants, task labels and filenames.
    """

    # create image labels based on lena's 6x6 grid
    grid_ticks = np.array([0.10, 1.06, 2.02, 2.98, 3.94, 4.90])
    grid_x, grid_y = np.meshgrid(grid_ticks, grid_ticks)
    all_grid_points = np.column_stack((grid_x.ravel(), grid_y.ravel()))

    center = 2.5 # center of shape space is the "boundary"
    all_quadrant = np.zeros([np.shape(all_grid_points)[0], 1])

    # assign each point to a quadrant based on its coordinates
    all_quadrant[np.logical_and(all_grid_points[:, 0] > center,
                                all_grid_points[:, 1] > center)] = 1
    all_quadrant[np.logical_and(all_grid_points[:, 0] < center,
                                all_grid_points[:, 1] > center)] = 2
    all_quadrant[np.logical_and(all_grid_points[:, 0] < center,
                                all_grid_points[:, 1] < center)] = 3
    all_quadrant[np.logical_and(all_grid_points[:, 0] > center,
                                all_grid_points[:, 1] < center)] = 4

    # task1: group quadrants 1+4 vs. 2+3
    labels_task1 = np.zeros(np.shape(all_quadrant))
    labels_task1[np.isin(all_quadrant, [1, 4])] = 2
    labels_task1[np.isin(all_quadrant, [2, 3])] = 1

    # task2: group quadrants 1+2 vs. 3+4
    labels_task2 = np.zeros(np.shape(all_quadrant))
    labels_task2[np.isin(all_quadrant, [1, 2])] = 2
    labels_task2[np.isin(all_quadrant, [3, 4])] = 1

    # task3: group quadrants 1+3 vs. 2+4 (nonlinear)
    labels_task3 = np.zeros(np.shape(all_quadrant))
    labels_task3[np.isin(all_quadrant, [1, 3])] = 2
    labels_task3[np.isin(all_quadrant, [2, 4])] = 1

    # full path to each blob image in the grid
    filenames_full = [
        os.path.join(image_dir, "blob_%.2f_%.2f.png" % (x, y))
        for x, y in zip(all_grid_points[:, 0], all_grid_points[:, 1])
    ]

    # assemble into a dataframe
    shape_df = pd.DataFrame.from_dict({
        "coord_axis1": all_grid_points[:, 0],
        "coord_axis2": all_grid_points[:, 1],
        "quadrant": np.squeeze(all_quadrant),
        "labels_task1": np.squeeze(labels_task1),
        "labels_task2": np.squeeze(labels_task2),
        "labels_task3": np.squeeze(labels_task3),
        "filename_full": filenames_full
    })

    return shape_df


def get_main_grid():
    """
    get coordinates for the 4x4 "main grid" used for separability analyses.
    """

    nsteps_main = 4 # 4 points along each axis for the main grid

    start_grid = 0.1
    stop_grid = 4.9

    main_pts = np.round(np.linspace(start_grid, stop_grid, nsteps_main), 1)
    gridx, gridy = np.meshgrid(main_pts, main_pts)
    main_grid_points = np.array([gridx.ravel(), gridy.ravel()]).T

    return main_grid_points


def get_category_separability(features,
                              labels):
    """
    compute henderson-style category separability for features and binary labels.
    """

    labels = np.array(labels).astype(int)
    assert np.array_equal(np.unique(labels), np.array([1, 2]))

    # split into the two category groups
    g1 = features[labels == 1, :]
    g2 = features[labels == 2, :]

    # within-category distances for each group separately
    within1 = scipy.spatial.distance.pdist(g1, metric="euclidean")
    within2 = scipy.spatial.distance.pdist(g2, metric="euclidean")

    # concatenate within distances across both categories
    within = np.concatenate([within1, within2], axis=0)

    # between-category distances (all pairs g1 vs g2)
    between = scipy.spatial.distance.cdist(g1, g2, metric="euclidean").ravel()

    b = np.mean(between)
    w = np.mean(within)

    # separability index (b - w) / (b + w)
    sep = (b - w) / (b + w)

    return sep


#%% step 1: make the image brick for matlab

# create the hdf5 file that matlab's get_gist.m will use
prep_brick(debug=False)

# sanity check to make sure the brick is correct (has the image data)
fn = os.path.join(feat_dir, "images_all.h5py")
print(fn)

with h5py.File(fn, "r") as f:
    print("keys:", list(f.keys()))
    stim = f["stimuli"][:] # load into memory

print("stimuli shape:", stim.shape)
print("dtype:", stim.dtype)
print("min, max:", np.min(stim), np.max(stim))

# show all images in a grid for visual inspection
n_images = stim.shape[0]
n_rows = int(np.ceil(np.sqrt(n_images)))
n_cols = int(np.ceil(n_images / n_rows))

plt.figure(figsize=(8, 8))

for i in range(n_images):
    ax = plt.subplot(n_rows, n_cols, i + 1)
    ax.imshow(stim[i, 0, :, :], cmap="gray")
    ax.axis("off")

plt.tight_layout()
plt.show()


#%% step 2: run the get_gist.m matlab script and download simclr

# must run the matlab gist pipeline before continuing to the next cell
# download the following file, place under the simclr dir:
#   https://github.com/Spijkervet/SimCLR/releases/download/1.2/checkpoint_100.tar


#%% step 3: extract simclr features with resnet

# define stuff about the resnet layers here
n_features_each_resnet_block = [
    256, 256, 256,
    512, 512, 512, 512,
    1024, 1024, 1024, 1024, 1024, 1024,
    2048, 2048, 2048
]
resnet_block_names = ["block%d" % nn for nn in range(len(n_features_each_resnet_block))]

# run feature extraction + pca for the new blob images
get_resnet_features(debug=False, training_type="simclr", image_set_name="images")

simclr_feat_dir = os.path.join(feat_dir, "simclr")

# sanity check that simclr pca files were created and that shapes look correct
for bb in [2, 6, 12, 15]:
    fn = os.path.join(simclr_feat_dir, f"images_simclr_block{bb}_pca.npy")
    print(fn, "exists:", os.path.exists(fn))
    if os.path.exists(fn):
        arr = np.load(fn)
        print("  shape:", arr.shape)


#%% step 4: compute category separability and bar plot

# load gist features (computed in matlab)
fn = os.path.join(feat_dir, "gist", "images_gistdescriptors_4ori_4blocks.h5py")
print("loading gist from:", fn)

with h5py.File(fn, "r") as file:
    f = np.array(file["/features"])

f_gist = f
print("f_gist shape:", f_gist.shape)

# load simclr features (blocks 2, 6, 12, 15) and concatenate across blocks
blocks_to_do = [2, 6, 12, 15]
f = []

for bb in blocks_to_do:
    fn = os.path.join(feat_dir, "simclr", "images_simclr_block%d_pca.npy" % bb)
    print("loading simclr block %d from: %s" % (bb, fn))
    ftmp = np.load(fn)
    f.append(ftmp)

f = np.concatenate(f, axis=1)
f_simclr = f
print("f_simclr shape:", f_simclr.shape)

# load labels + get 6x6 main grid indices (for separability)
labs = pd.read_csv(os.path.join(feat_dir, "Image_labels_grid3.csv"))

# coordinates for each point in the full grid
pts = np.array([labs["coord_axis1"], labs["coord_axis2"]]).T

# main grid coordinates (4x4)
main_grid = get_main_grid()

# find closest index in the full 6x6 grid for each main-grid point
gi = [np.argmin(np.sum((pts - g) ** 2, axis=1)) for g in main_grid]
gi = np.array(gi, dtype=int)
print("number of main grid points:", gi.size)

# subset to main grid for separability
f_gist_main = f_gist[gi, :]
f_simclr_main = f_simclr[gi, :]

# compute separability per task axis for gist and simclr
gist_means = np.zeros((3,))
simclr_means = np.zeros((3,))
gist_sem = np.zeros((3,))
simclr_sem = np.zeros((3,))

for ai, axis in enumerate([1, 2, 3]):

    # grab labels for this axis and restrict to main grid indices
    l = np.array(labs["labels_task%d" % axis])[gi].astype(int)

    # category separability for gist and simclr
    m_gist = get_category_separability(f_gist_main, l)
    m_sim = get_category_separability(f_simclr_main, l)

    gist_means[ai] = m_gist
    simclr_means[ai] = m_sim

print("gist_means:", gist_means)
print("simclr_means:", simclr_means)

# set font embedding so pdfs behave nicely
plt.rcParams["pdf.fonttype"] = 42

task_names = ["Linear (1)", "Linear (2)", "Nonlinear", "Repeat"]

# color palette similar to henderson et al. (light to dark blues)
task_colors = np.flipud(cm.GnBu(np.linspace(0, 1, 5))[1:, :])

# vals is [3 tasks x 2 models] (gist, simclr)
vals = np.array([gist_means, simclr_means]).T

# position the bars so each task is slightly jittered around each model
xjitter = np.linspace(-0.25, 0.25, 3)
bw = np.diff(xjitter)[0] / 4 * 3

plt.figure(figsize=(4, 4))
ax = plt.subplot(1, 1, 1)

error_kw = dict(lw=0.8, capsize=3, capthick=0.8, ecolor="k")

# loop over models (gist vs simclr) and draw bars for each task
for xi in range(2):

    yvals = vals[:, xi]
    plt.bar(
        xi + np.array(xjitter),
        yvals,
        width=bw,
        color=task_colors[0:3, :],
        error_kw=error_kw
    )

plt.xticks(np.arange(2), ["GIST", "SimCLR"])
plt.ylabel("Category separability (a.u.)", fontsize=16)
plt.yticks([0, 0.05, 0.10, 0.15, 0.20], fontsize=20) # hard-coding these to match henderson et al.
plt.xticks(fontsize=20)

# visually match y-axis range used by henderson et al.
plt.ylim([0, 0.21])

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# legend for task colors (linear1, linear2, nonlinear)
handles = [plt.Rectangle((0, 0), 1, 1, color=task_colors[i, :]) for i in range(3)]
ax.legend(
    handles,
    task_names[0:3],
    title="Task",
    frameon=False,
    loc="upper left",
    bbox_to_anchor=(1.02, 1.0),
    borderaxespad=0.0
)

figname = os.path.join(image_sim_dir, "image_category_sep.png")
plt.savefig(figname, dpi=600, bbox_inches="tight")
plt.show()
print("saved:", figname)

# keep around a small source dataframe with the values used for plotting
source_df = pd.DataFrame(vals, index=task_names[0:3], columns=["GIST", "SimCLR"])


#%% step 5: pca on gist features and scatter plots

# run pca on gist features (all 36 blobs in the main grid)
scores, wts, pre_mean, ev = compute_pca(f_gist, max_pc_to_retain=100, copy_data=True)
print("scores shape:", scores.shape)

scores_all = scores

# pull axis coordinates from labels for coloring
axis1_vals = np.array(labs["coord_axis1"])
axis2_vals = np.array(labs["coord_axis2"])

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

# panel 1: points colored by dimension 1 coordinate
for ii in range(scores_all.shape[0]):
    ax1.plot(
        scores_all[ii, 0],
        scores_all[ii, 1],
        "o",
        color=cmap(norm1(axis1_vals[ii]))
    )

ax1.set_aspect("equal", "box")
ax1.set_xticks([])
ax1.set_yticks([])
ax1.set_xlabel("PC 1", fontsize=20)
ax1.set_ylabel("PC 2", fontsize=20)
ax1.spines["top"].set_visible(False)
ax1.spines["right"].set_visible(False)

# panel 2: points colored by dimension 2 coordinate
for ii in range(scores_all.shape[0]):
    ax2.plot(
        scores_all[ii, 0],
        scores_all[ii, 1],
        "o",
        color=cmap(norm2(axis2_vals[ii]))
    )

ax2.set_aspect("equal", "box")
ax2.set_xticks([])
ax2.set_yticks([])
ax2.set_xlabel("PC 1", fontsize=20)
ax2.set_ylabel("PC 2", fontsize=20)
ax2.spines["top"].set_visible(False)
ax2.spines["right"].set_visible(False)

# colorbar for axis 1 coords
cb1 = plt.colorbar(
    cm.ScalarMappable(norm=norm1, cmap=cmap),
    cax=cax1,
    orientation="horizontal"
)
cb1.set_label("axis 1 coordinate (arb. units)", fontsize=14)
cb1.set_ticks([])
cb1.set_ticklabels([])
cax1.tick_params(labelsize=9)

# colorbar for axis 2 coords
cb2 = plt.colorbar(
    cm.ScalarMappable(norm=norm2, cmap=cmap),
    cax=cax2,
    orientation="horizontal"
)
cb2.set_label("axis 2 coordinate (arb. units)", fontsize=14)
cb2.set_ticks([])
cb2.set_ticklabels([])
cax2.tick_params(labelsize=9)

figname = os.path.join(image_sim_dir, "image_gist_pca.png")
plt.savefig(figname, dpi=600, bbox_inches="tight")
plt.show()
print("saved:", figname)