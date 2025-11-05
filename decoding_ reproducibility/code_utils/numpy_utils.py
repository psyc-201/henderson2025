import numpy as np
import pandas as pd
from itertools import product

def bin_ydata_by_xdata(xdata, ydata, n_bins, linear_bins=True, remove_nans=True, \
                       return_edges=False, return_std = False, use_unique=False):
           
    if len(xdata.shape)>1:
        xdata = np.squeeze(xdata)
    if len(ydata.shape)>1:
        ydata = np.squeeze(ydata)
    assert((len(xdata.shape)==1) and (len(ydata.shape)==1))
    
    if linear_bins:
        # break x axis into linearly spaced increments
        bin_edges = np.linspace(np.min(xdata), np.max(xdata)+0.0001, n_bins+1)
        bin_centers = bin_edges[0:-1]+(bin_edges[1]-bin_edges[0])/2
    elif use_unique:
        # force the bin centers to take on actual values in the data.
        bin_centers = np.unique(xdata)
        bin_edges = np.concatenate([bin_centers-0.01, [bin_centers[-1]+0.01]], axis=0)
        n_bins=len(bin_centers)
    else:
        # bin according to data density
        bin_edges = np.quantile(xdata, np.linspace(0,1,n_bins+1))
        bin_edges[-1]+=0.0001
        bin_centers = bin_edges[0:-1]+np.diff(bin_edges)/2

    xbinned = np.zeros((n_bins,))
    ybinned = np.zeros((n_bins,))
    ybin_std = np.zeros((n_bins,))
    used_yet = np.zeros((len(xdata),))
    for bb in range(n_bins):
        inds = (xdata>=bin_edges[bb]) & (xdata<bin_edges[bb+1])
        xbinned[bb] = bin_centers[bb]
        if np.sum(inds)>0:
            used_yet[inds] += 1
            ybinned[bb] = np.mean(ydata[inds])
            ybin_std[bb] = np.std(ydata[inds])
        else:
            ybinned[bb] = np.nan
            ybin_std[bb] = np.nan
            
    assert(np.all(used_yet)==1)
   
    if remove_nans:
        good = ~np.isnan(ybinned)
        xbinned=xbinned[good] 
        bin_edges=np.concatenate([bin_edges[0:-1][good], [bin_edges[-1]]], axis=0)
        ybinned=ybinned[good]
        ybin_std=ybin_std[good]
        
    to_return = xbinned, ybinned
    
    if return_edges:
        to_return += bin_edges,
    
    if return_std:
        to_return += ybin_std,
        
    return to_return


def bin_vals(vals, bin_edges):
    
    # bin vals according to bins in bin_edges
    # bin edges must not be equal to any values in vals
    
    assert(not np.any(np.isin(bin_edges, vals)))
    
    vals_binned = np.zeros(vals.shape, dtype=int)-1
    n_bins = len(bin_edges)-1
    
    for cb in np.arange(n_bins):
        binds = (vals>=bin_edges[cb]) & (vals<=bin_edges[cb+1])
        assert(np.all(vals_binned[binds]==-1))
        vals_binned[binds] = cb
        
    return vals_binned



def array_to_dataframe(array, dim_labels):
    # Get the shape of the array
    shape = array.shape
    
    # Generate all possible index combinations
    indices = list(product(*[range(s) for s in shape]))
    
    # Flatten the array
    flat_data = array.flatten()
    
    # Create a DataFrame
    df = pd.DataFrame(indices, columns=dim_labels)
    df['value'] = flat_data
    
    return df

