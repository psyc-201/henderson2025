import numpy as np
import scipy.stats
import warnings
import pandas as pd
import statsmodels
import statsmodels.stats.multitest
from statsmodels.stats.anova import AnovaRM


def lin_reg(x,y):
   
    if len(x.shape)==1:
        x_mat = x[:,np.newaxis]
    else:
        x_mat = x
    if len(y.shape)==1:
        y_mat = y[:,np.newaxis]
    else:
        y_mat = y
        
    n_pts = x_mat.shape[0]
    assert(y_mat.shape[0]==n_pts)
    
    X = np.concatenate([x_mat, np.ones((n_pts,1))], axis=1)
    reg_coeffs = np.linalg.pinv(X) @ y_mat
    yhat = X @ reg_coeffs
    
    actual = np.squeeze(y_mat)
    pred = np.squeeze(yhat)
    ssres = np.sum(np.power((actual - pred),2));
    sstot = np.sum(np.power((actual - np.mean(actual)),2));
    r2 = 1-(ssres/sstot)
    
    return yhat, reg_coeffs, r2

def get_dprime(predlabs,reallabs,un=None):
    """ 
    Calculate d' for predicted and actual values. Works for multiple classes.
    """

    predlabs==np.squeeze(predlabs)
    reallabs==np.squeeze(reallabs)
    if len(predlabs)!=len(reallabs):
        raise ValueError('real and predicted labels do not match')
    if len(predlabs.shape)>1 or len(reallabs.shape)>1:
        raise ValueError('need to have 1d inputs')
    if un is None:
        un = np.unique(reallabs)
    if not np.all(np.isin(np.unique(predlabs), un)):
        print('Warning: some labels in pred are not included in real labels! Will return nan')
        return np.nan
    
    hrz=np.zeros((len(un),1));
    fpz=np.zeros((len(un),1));

    n_trials = len(predlabs);

    #loop over class labels, get a hit rate and false pos for each (treating
    #any other category as non-hit)
    for ii in range(len(un)):

        if np.sum(reallabs==un[ii])==0 or np.sum(reallabs!=un[ii])==0:

            # if one of the categories is completely absent - this will return a
            # nan dprime value
            return np.nan

        else:

            hr = np.sum((predlabs==un[ii]) & (reallabs==un[ii]))/np.sum(reallabs==un[ii]);
            fp = np.sum((predlabs==un[ii]) & (reallabs!=un[ii]))/np.sum(reallabs!=un[ii]);    

            # make sure this never ends up infinite
            # correction from Macmillan & Creelman, use 1-1/2N or 1/2N in place
            # of 1 or 0 
            if hr==0:
                hr=1/(2*n_trials)
            if fp==0:
                fp=1/(2*n_trials)
            if hr==1:
                hr=1-1/(2*n_trials)
            if fp==1:
                fp=1-1/(2*n_trials);

        # convert to z score (this is like percentile - so 50% hr would be zscore=0)
        hrz[ii]=scipy.stats.norm.ppf(hr,0,1);
        fpz[ii]=scipy.stats.norm.ppf(fp,0,1);

    # dprime is the mean of individual dprimes (for two classes, they will be
    # same value)
    dprime = np.mean(hrz-fpz);

    return dprime




def compute_partial_corr(x, y, c, return_p=False):

    """
    Compute the partial correlation coefficient between x and y, 
    controlling for the variables in covariates "c". 
    Uses linear regression based method.
    Inputs: 
        x [n_samples,] or [n_samples,1]
        y [n_samples,] or [n_samples,1]
        c [n_samples,] or [n_samples,n_covariates]
        
    Outputs:
        partial_corr, a single value for the partial correlation coefficient.
    """
    
    if len(x.shape)==1:
        x = x[:,np.newaxis]        
    if len(y.shape)==1:
        y = y[:,np.newaxis]
    if len(c.shape)==1:
        c = c[:,np.newaxis]
    n_trials = x.shape[0]
    assert(y.shape[0]==n_trials and c.shape[0]==n_trials)
    
    # first predict x from the other vars
    model1_preds = np.concatenate([c, np.ones((n_trials,1))], axis=1)
    model1_coeffs = np.linalg.pinv(model1_preds) @ x
    model1_yhat = model1_preds @ model1_coeffs
    model1_resids = model1_yhat - x
   
    # then predict y from the other vars
    model2_preds = np.concatenate([c, np.ones((n_trials,1))], axis=1)
    model2_coeffs = np.linalg.pinv(model2_preds) @ y
    model2_yhat = model2_preds @ model2_coeffs
    model2_resids = model2_yhat - y

    # correlate the residuals to get partial correlation.
    if return_p:
        partial_corr, p = scipy.stats.pearsonr(model1_resids[:,0], model2_resids[:,0])
        return partial_corr, p
    else:
        partial_corr = numpy_corrcoef_warn(model1_resids[:,0], model2_resids[:,0])[0,1]
        return partial_corr
   
    

def compute_partial_corr_formula(x,y,c):
 
    """
    Code to compute the partial correlation between x and y, controlling
    for covariate c. 
    Based on the correlation coefficients between each pair of variables.
    Also computes estimated standardized beta weight. 
    """
    x = np.squeeze(x); 
    y = np.squeeze(y);
    c = np.squeeze(c);
    ryx = np.corrcoef(x,y)[0,1]
    ryc = np.corrcoef(c,y)[0,1]
    rxc = np.corrcoef(x,c)[0,1]

    # partial correlation coefficient
    partial_corr = (ryx - ryc*rxc)/np.sqrt((1-ryc**2)*(1-rxc**2))
  
    # equivalent to standardized beta weight from a multiple linear regression
    # would be set up like [x, c, intercept] @ w = y
    # this is the weight for x.
    # standarized beta = raw beta * std(x)/std(y)
    beta = (ryx - ryc*rxc)/(1-rxc**2)
    
    return partial_corr, beta

# Some functions that wrap basic numpy/scipy functions, but will print 
# more useful warnings when a problem arises

def numpy_corrcoef_warn(a,b):
    
    with warnings.catch_warnings():
        warnings.filterwarnings('error')
        try:
            cc = np.corrcoef(a,b)
        except RuntimeWarning as e:
            print('Warning: problem computing correlation coefficient')
            print('shape a: ',a.shape)
            print('shape b: ',b.shape)
            print('sum a: %.9f'%np.sum(a))
            print('sum b: %.9f'%np.sum(b))
            print('std a: %.9f'%np.std(a))
            print('std b: %.9f'%np.std(b))
            print(e)
            warnings.filterwarnings('ignore')
            cc = np.corrcoef(a,b)
            
    if np.any(np.isnan(cc)):
        print('There are nans in correlation coefficient')
    
    return cc

def paired_ttest_nonpar(vals1, vals2, n_iter=1000, rndseed=None):
    
    if rndseed is None:
        rndseed = int(time.strftime('%M%H%d', time.localtime()))
    np.random.seed(rndseed)
        
    d = vals1 - vals2
    n = len(d)
    
    # this is how the paired t-stat is computed in scipy.stats.ttest_rel
    v = np.var(d)*n/(n-1)
    denom = np.sqrt(v/n)
    real_t = np.mean(d)/denom
 
    shuff_t = np.zeros((n_iter,))
    
    for ii in range(n_iter):
        
        shuff_vals = np.array([vals1, vals2])
        # randomly swap the positions of values within a pair, with 50% prob
        which_swap = np.random.normal(0,1,[len(vals1),])>0
        shuff_vals[:,which_swap] = np.flipud(shuff_vals[:, which_swap])
    
        shuff_d = shuff_vals[0,:] - shuff_vals[1,:]
        v = np.var(shuff_d)*n/(n-1)
        denom = np.sqrt(v/n)
        shuff_t[ii] = np.mean(shuff_d)/denom
      
    # pvalue for two-tailed test
    pval_twotailed = np.minimum( np.mean(shuff_t<=real_t), \
                                 np.mean(shuff_t>=real_t)) * 2
    
    return pval_twotailed, real_t



def rmanova_1way(dat, dim_name, do_shuffle=False, n_iter=1000, rndseed=None):
    
    # dat is [subjects x nlevels1] 
    
    n_subjects, n_levels1 = dat.shape
    
    dim1 = dim_name
    
    class_df = pd.DataFrame(data={'subject': np.repeat(np.arange(n_subjects), n_levels1), \
                                   dim1: np.tile(np.arange(n_levels1),[n_subjects,])})

    class_values = np.zeros((n_subjects*n_levels1))
    for si in range(n_subjects):
        inds = (class_df['subject']==si)
        class_values[inds] = dat[si,:]
    class_df['class_values'] = class_values

    model = AnovaRM(data=class_df, \
                    depvar='class_values', \
                    subject='subject', \
                    within = [dim1],
                   )
    
    rm_result = model.fit()
    
    anova_table = rm_result.anova_table
    
    if do_shuffle:
        
        real_f = np.array(rm_result.anova_table['F Value'])
        
        shuff_f = np.zeros((n_iter,1))

        if rndseed is not None:
            np.random.seed(rndseed)
    
        for xx in range(n_iter):
            
            shuff_df = class_df.copy(deep=True)
            orig_v = np.array(shuff_df['class_values'])
            shuff_v = np.zeros_like(orig_v)

            # shuffle all vals within each subject
            for si in range(n_subjects):
                sidx = np.array(shuff_df['subject']==si)
                v = orig_v[sidx]
                shuff_v[sidx] = v[np.random.permutation(len(v))]

            shuff_df['class_values'] = shuff_v

            shuff_model = AnovaRM(data=shuff_df, \
                        depvar='class_values', \
                        subject='subject', \
                        within = [dim1],
                       )
            
            shuff_rm_result = shuff_model.fit()
            shuff_f[xx,:] = np.array(shuff_rm_result.anova_table['F Value'])

        p = np.mean(shuff_f>=real_f, axis=0)

        # add permuted p as another column in table
        anova_table['p (permutation)'] = p
       
    
    return anova_table



def rmanova_2way(dat, dim_names, do_shuffle=False, n_iter=1000, rndseed=None):
    
    # dat is [subjects x nlevels1 x nlevels2] 
    
    n_subjects, n_levels1, n_levels2 = dat.shape
    
    dim1 = dim_names[0]
    dim2 = dim_names[1]
    
    class_df = pd.DataFrame(data={'subject': np.repeat(np.arange(n_subjects), n_levels1*n_levels2), \
                                   dim1: np.tile(np.repeat(np.arange(n_levels1), n_levels2),[n_subjects,]), \
                                   dim2: np.tile(np.arange(n_levels2), [n_subjects*n_levels1,])})

    class_values = np.zeros((n_subjects*n_levels1*n_levels2))
    for si in range(n_subjects):
        for li in range(n_levels1):
            inds = (class_df['subject']==si) & (class_df[dim1]==li)
            class_values[inds] = dat[si,li,:]
    class_df['class_values'] = class_values

    model = AnovaRM(data=class_df, \
                    depvar='class_values', \
                    subject='subject', \
                    within = [dim1, dim2],
                   )
    
    rm_result = model.fit()
    
    anova_table = rm_result.anova_table
    
    
    if do_shuffle:
        
        real_f = np.array(rm_result.anova_table['F Value'])
        
        shuff_f = np.zeros((n_iter,3))

        if rndseed is not None:
            np.random.seed(rndseed)
    
        for xx in range(n_iter):
            
            shuff_df = class_df.copy(deep=True)
            orig_v = np.array(shuff_df['class_values'])
            shuff_v = np.zeros_like(orig_v)

            # shuffle all vals within each subject
            for si in range(n_subjects):
                sidx = np.array(shuff_df['subject']==si)
                v = orig_v[sidx]
                shuff_v[sidx] = v[np.random.permutation(len(v))]

            shuff_df['class_values'] = shuff_v

            shuff_model = AnovaRM(data=shuff_df, \
                        depvar='class_values', \
                        subject='subject', \
                        within = [dim1, dim2],
                       )
            
            shuff_rm_result = shuff_model.fit()
            shuff_f[xx,:] = np.array(shuff_rm_result.anova_table['F Value'])

        p = np.mean(shuff_f>=real_f, axis=0)

        # add permuted p as another column in table
        anova_table['p (permutation)'] = p
       
    
    return anova_table






def rmanova_3way(dat, dim_names, do_shuffle=False, n_iter=1000, rndseed=None):
    
    # dat is [subjects x nlevels1 x nlevels2] 
    
    n_subjects, n_levels1, n_levels2, n_levels3 = dat.shape
    
    dim1 = dim_names[0]
    dim2 = dim_names[1]
    dim3 = dim_names[2]
    
    class_df = pd.DataFrame(data={'subject': np.repeat(np.arange(n_subjects), n_levels1*n_levels2*n_levels3), \
                               dim1: np.tile(np.repeat(np.arange(n_levels1), n_levels2*n_levels3),[n_subjects,]), \
                               dim2: np.tile(np.repeat(np.arange(n_levels2), n_levels3), [n_subjects*n_levels1,]), \
                               dim3: np.tile(np.arange(n_levels3), [n_subjects*n_levels1*n_levels2,])})

    
    class_values = np.zeros((n_subjects*n_levels1*n_levels2*n_levels3,))
    for si in range(n_subjects):
        for li1 in range(n_levels1):
            for li2 in range(n_levels2):
                inds = (class_df['subject']==si) & (class_df[dim1]==li1) & (class_df[dim2]==li2)
                class_values[inds] = dat[si,li1,li2,:]
                
    class_df['class_values'] = class_values

    model = AnovaRM(data=class_df, \
                    depvar='class_values', \
                    subject='subject', \
                    within = [dim1, dim2, dim3],
                   )
    
    rm_result = model.fit()
    
    anova_table = rm_result.anova_table
    
    
    if do_shuffle:
        
        real_f = np.array(rm_result.anova_table['F Value'])
        
        shuff_f = np.zeros((n_iter,7))

        if rndseed is not None:
            np.random.seed(rndseed)
    
        for xx in range(n_iter):
            
            shuff_df = class_df.copy(deep=True)
            orig_v = np.array(shuff_df['class_values'])
            shuff_v = np.zeros_like(orig_v)

            # shuffle all vals within each subject
            for si in range(n_subjects):
                sidx = np.array(shuff_df['subject']==si)
                v = orig_v[sidx]
                shuff_v[sidx] = v[np.random.permutation(len(v))]

            shuff_df['class_values'] = shuff_v

            shuff_model = AnovaRM(data=shuff_df, \
                        depvar='class_values', \
                        subject='subject', \
                        within = [dim1, dim2, dim3],
                       )
            
            shuff_rm_result = shuff_model.fit()
            shuff_f[xx,:] = np.array(shuff_rm_result.anova_table['F Value'])

        p = np.mean(shuff_f>=real_f, axis=0)

        # add permuted p as another column in table
        anova_table['p (permutation)'] = p
       
    
    return anova_table



def rmanova_4way(dat, dim_names, do_shuffle=False, n_iter=1000, rndseed=None):
    
    # dat is [subjects x nlevels1 x nlevels2 x nlevels3] 

    n_subjects, n_levels1, n_levels2, n_levels3, n_levels4 = dat.shape

    dim1 = dim_names[0]
    dim2 = dim_names[1]
    dim3 = dim_names[2]
    dim4 = dim_names[3]

    class_df = pd.DataFrame(data={'subject': np.repeat(np.arange(n_subjects), n_levels1*n_levels2*n_levels3*n_levels4), \
                               dim1: np.tile(np.repeat(np.arange(n_levels1), n_levels2*n_levels3*n_levels4),[n_subjects,]), \
                               dim2: np.tile(np.repeat(np.arange(n_levels2), n_levels3*n_levels4), [n_subjects*n_levels1,]), \
                               dim3: np.tile(np.repeat(np.arange(n_levels3), n_levels4), [n_subjects*n_levels1*n_levels2,]), \
                               dim4: np.tile(np.arange(n_levels4), [n_subjects*n_levels1*n_levels2*n_levels3,]) })


    class_values = np.zeros((n_subjects*n_levels1*n_levels2*n_levels3*n_levels4,))
    for si in range(n_subjects):
        for li1 in range(n_levels1):
            for li2 in range(n_levels2):
                for li3 in range(n_levels3):
                    inds = (class_df['subject']==si) & (class_df[dim1]==li1) & (class_df[dim2]==li2) & (class_df[dim3]==li3)
                    class_values[inds] = dat[si,li1,li2,li3,:]

    class_df['class_values'] = class_values

    model = AnovaRM(data=class_df, \
                    depvar='class_values', \
                    subject='subject', \
                    within = [dim1, dim2, dim3, dim4],
                   )

    rm_result = model.fit()

    anova_table = rm_result.anova_table

    if do_shuffle:
        
        real_f = np.array(rm_result.anova_table['F Value'])
        
        shuff_f = np.zeros((n_iter,15))

        if rndseed is not None:
            np.random.seed(rndseed)
    
        for xx in range(n_iter):
            
            shuff_df = class_df.copy(deep=True)
            orig_v = np.array(shuff_df['class_values'])
            shuff_v = np.zeros_like(orig_v)

            # shuffle all vals within each subject
            for si in range(n_subjects):
                sidx = np.array(shuff_df['subject']==si)
                v = orig_v[sidx]
                shuff_v[sidx] = v[np.random.permutation(len(v))]

            shuff_df['class_values'] = shuff_v

            shuff_model = AnovaRM(data=shuff_df, \
                        depvar='class_values', \
                        subject='subject', \
                        within = [dim1, dim2, dim3, dim4],
                       )
            
            shuff_rm_result = shuff_model.fit()
            shuff_f[xx,:] = np.array(shuff_rm_result.anova_table['F Value'])

        p = np.mean(shuff_f>=real_f, axis=0)

        # add permuted p as another column in table
        anova_table['p (permutation)'] = p
       
    
    return anova_table


def fdr_keepshape(pvals, alpha=0.05, method='indep'):
    
    """
    This is a wrapper for the fdr function in statsmodels, allows
    for entering a 2D array and FDR correct all values together.
    Returns arrays same shape as original.
    """
    orig_shape = pvals.shape
    pvals_reshaped = pvals.ravel()
    
    pvals_fdr, masked_fdr = statsmodels.stats.multitest.fdrcorrection(pvals_reshaped, alpha=alpha, method=method)
    
    pvals_fdr = np.reshape(pvals_fdr, orig_shape)
    masked_fdr = np.reshape(masked_fdr, orig_shape)
    
    return pvals_fdr, masked_fdr