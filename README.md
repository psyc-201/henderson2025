# Replication Project: Henderson et al. (2025)
This repository reproduces the ROI × Task × Boundary decoding analysis (near-trial ANOVA) reported in Henderson et al. (2025). It also re-creates Figure 2A–C to check whether the qualitative pattern of binary decoding accuracies emerges in an independent implementation.
Beyond replication, the repository generates a new set of silhouette stimuli and evaluates their structure, including dimensionality and category separability under multiple decision boundaries.

## Recreating the environment
The file `henderson2025_env.yml` specifies the full conda + pip environment used to run all analyses. After cloning the repository (see steps below), build this environment on your machine:

```
conda env create -f henderson2025_env.yml
conda activate henderson2025_env
```
All Matlab scripts utilized MATLAB_R2024b.

## How to Run the Reproducibility Test
1. Clone the repository into whatever directory of your choosing, and then navigate into it:

```
git clone https://github.com/psyc-201/henderson2025
cd henderson2025
```

2. Create a folder called `data` inside the top layer of the repo directory.  

3. Download the data files from the authors’ [OSF repo](https://osf.io/fa8jk/) and move them into the appropriate folders:
- Specifically, download the `behav_all.zip`, unzip it, then place this directory within `data`
- To download the voxel activation and timing data, download all files with the the author's `Samples` directory on OSF.
- You may have to download these `Samples` files one-by-one then move them ALL into a folder called `Samples` that you make yourself (the zip file seemed to have issues on OSF)
- Both `Samples` and `DataBehavior` should be nested within `data` 

4. Open the file `decoding_reproducibility/reproducing_decoding_anova_result.py` in an IDE. Make sure you have built and activated the Python environment.

5. Run the script.  
It will:
- Load in the data from the .mat files and convert them into a Python format,
- Perform decoding within each ROI,
- Run a 3-way repeated-measures ANOVA (ROI × Task × Boundary),
- Save figures and results to the `bold_decoding_anova_results/` folder. All outputted PNGs and CSVs are also available on on [OSF](https://osf.io/4p63n/files/osfstorage).

## How to Run the Simulation/Extension w/ New Stimuli
This extension pipeline recreates the representational analyses from Henderson et al. (2025) using a new set of silhouette stimuli. The Python script is organized into clear `#%% step blocks`, but you must begin by generating the new stimulus images in MATLAB.

1. Generate the new silhouettes in MATLAB
- Open the script `stimuli/making_new_blobs.m` in MATLAB.
- Open the script `stimuli/making_new_blobs.m` in MATLAB.
- Confirm that it writes 36 PNG files into the folder: `stimuli/new_blob_stim/` These images will be used by all later steps.

2. Build the HDF5 image brick (Python)
- Open `image_similarity/image_similarity.py` in your IDE.
- Run the cell labeled: #`%% step 1: make the image brick for matlab`
- This step loads the PNGs and writes a single HDF5 file here: `image_similarity/features/images_all.h5py`

3. Run MATLAB GIST and prepare SimCLR weights
- Run the MATLAB GIST extraction script `image_similarity/gist_matlab/get_gist.m` (uses the the HDF5 image brick we created). This will produce the file: `image_similarity/features/gist/images_gistdescriptors_4ori_4blocks.h5py`
- Download the SimCLR checkpoint file `checkpoint_100.tar` from the [SimCLR GitHub releases page](https://pypi.org/project/simclr/).
- Place the checkpoint inside: `image_similarity/features/simclr/` The Python feature extraction will not run until this file is present.

4. Extract SimCLR features (Python)
- In `image_similarity.py`, run the cell labeled: `#%% step 3: extract simclr features with resnet`
- This step computes PCA-reduced features for blocks 2, 6, 12, and 15 and saves files such as: `image_similarity/features/simclr/images_simclr_block2_pca.npy`

5. Compute category separability
- Run the cell labeled: `#%% step 4: compute category separability and bar plot`
- This step loads the GIST and SimCLR features, computes category separability for each decision boundary, and produces the figure: `image_similarity/image_category_sep.png`

6. PCA visualization
- Run the cell labeled: `#%% step 5: pca on gist features and scatter plots`
- This produces a 2-panel PCA visualization saved as: `image_similarity/image_gist_pca.png`

## Overview of directories
- `writeup` - contains Quarto markdown file and rendered html report. Contains a subdirectory, `figs`, that houses the figures used in the markdown report. These are available on [OSF](https://osf.io/4p63n/files/osfstorage) under the top-level `figs` directory
- `image_similarity` - contains the main Python script as well as the `gist_matlab` and `features` subdirectories. The produced PNGs from this pipeline are available [here](https://osf.io/4p63n/files/osfstorage) under the `image_similarity` directory. All intermediate files: e.g., h5py, mat, and CSV files, are also available on this OSF link under the subdirectories nested within `image_similarity`. Basically conyains all Gist and SimCLR implementation, along with `image_similarity.py`
- `stimuli` - contains 'making_new_blobs.m` and output directory for newly generated shape stimuli
- `post_hoc_power_analysis` - contains Python script to run simple post-hoc power analysis on the Task x Interaction effect of interest
- `original_paper` - contains PDF copy of the Henderson et al. (2025) paper
- `in_class_plotting_exercise` - contains Quarto markdown file to plot the reproduced Figure 2A-2C (decoding accuracy plots) in R
- `decoding_reproducibility` - contains `reproducing_decoding_anova_result.py`
