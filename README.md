# Replication Project: Henderson et al. (2025)
This repo reproduces the ROI × Task × Boundary decoding analysis (near-trial ANOVA) from Henderson et al. (2025). It also creates Figure 2A-C to examine the general trend of results (i.e. whether the means/general direction of the data stayed the same)

## How to Run the Reproducibility Test
1. Clone the repository into whatever directory of your choosing, and then navigate into it:

```
git clone https://github.com/psyc-201/henderson2025
cd henderson2025
```

2. Download the original authors’ helper function code (`code_utils`) from their [GitHub repo](https://github.com/mmhenderson/shapeDim) and place it in the top layer of the repo directory.

3. Create a folder called `data` inside this directory.  
Within `data`, make subfolders named `Samples` and `DataBehavior`.

4. Download the data files from the authors’ [OSF repo](https://osf.io/fa8jk/) and move them into the appropriate folders:
- `.mat` and `.csv` files go inside `data/Samples` and `data/DataBehavior`.
- You may have to download the `Samples` files one-by-one then move them into this folder (the zip file seemed to have issues on OSF)

5. Open the file `reproducing_decoding_anova_result.py` in an IDE.

6. Run the script.  
It will automatically:
- Load in the data from the .mat files and convert them into a Python format,
- Perform decoding within each ROI,
- Run a 3-way repeated-measures ANOVA (ROI × Task × Boundary),
- Save figures and results to the `bold_decoding_anova_results/` folder.
