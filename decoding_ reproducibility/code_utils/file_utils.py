import scipy.io as spio
import numpy as np
import os
import h5py

def load_samplefile_h5py(sample_fn):

    """
    This is a slightly hacky solution to load a SampleFile into python
    The SampleFile is made in matlab and saved as a .mat file in v7.3 format, 
    using MakeSampleFile.m.
    Because it is v7.3 we need to load using h5py instead of scipy.loadmat
    This code works for a specific set of keys (i.e. elements that are saved 
    into the matlab file) that are from this specific preprocessing pipeline
    Would need to be adapted for other preprocessing pipelines.
    """
    
    # going to make a dictionary for the elements in the file.
    # first extract these three keys of interest, these ones are 
    # relatively straightforward
    samples = {}
    keys_do = ['all_vox_concat', 'samplesMain', 'samplesRep']

    with h5py.File(sample_fn, 'r') as f:
        for kk in keys_do:
            # need to convert the element to an np.array
            samples[kk] = np.array(f[kk])

    # for the element "ROIs", it is a structure array so is more complicated.
    # going to make a separate dictionary
    rois_keys = ['voxel_inds', 'is_md','is_motor','is_visual']
    rois_dict = {}
    ROI_names = ['V1','V2','V3','V3AB','hV4','IPS0','IPS1','IPS2','IPS3','LO1','LO2'];
    hemis = ['lh', 'rh'];
    n_rois = 11; n_hemis = 2;

    # initializing a dictionary to store the info about ROI definitions.
    for rk in rois_keys:
        rois_dict[rk] = [[[] for hh in range(n_hemis)] for rr in range(n_rois)]
    
    all_vox_list = []

    with h5py.File(sample_fn, 'r') as f:

        # loop over the keys into the "ROIs" element here
        # (we know in advance what they are)
        for rk in rois_keys:
            # each element is [11,2] so loop over the arrays
            for rr in range(n_rois):
                for hh in range(n_hemis):

                    # this is the confusing part...
                    # ref will be a "reference" to the element that we're looking for
                    # like <HDF5 object reference>
                    ref = f['ROIs'][rk][rr][hh]
                
                    # to get the actual data, we have to use the reference as a key into the 
                    # original dataset (f), and then convert it into np array.
                    rois_dict[rk][rr][hh] = np.array(f[ref]).astype(int)

                    if rk=='voxel_inds':
                        a = np.array(f[ref]).astype(int)
                        if len(np.shape(a))==2:
                            all_vox_list += [a]
                        # else:
                        #     # skip any that are wrong shape, empty
                        #     print('missing voxel_inds for %s %s'%(ROI_names[rr], hemis[hh]))

    # double checking that the list of voxels we pulled out is correct and lines up with all_vox_concat
    # print([av.shape for av in all_vox_list])
    all_vox_list = np.concatenate(all_vox_list, axis=0)
    assert(len(np.unique(all_vox_list))==len(all_vox_list))
    assert(np.all(np.unique(all_vox_list)==samples['all_vox_concat'][:,0]))
    
    # put the ROIs into my main dict now, for simplicity
    samples['ROIs'] = rois_dict
    samples['ROI_names'] = ROI_names
    samples['hemis'] = hemis
    
    return samples

def load_locvalsfile_h5py(sample_fn):

    """
    Same as above, but for localizer t-stat vals only
    """
    
    # going to make a dictionary for the elements in the file.
    # first extract these three keys of interest, these ones are 
    # relatively straightforward
    samples = {}
    keys_do = ['all_vox_concat', 'loc_vals']

    with h5py.File(sample_fn, 'r') as f:
        for kk in keys_do:
            # need to convert the element to an np.array
            samples[kk] = np.array(f[kk])

    # for the element "ROIs", it is a structure array so is more complicated.
    # going to make a separate dictionary
    rois_keys = ['voxel_inds', 'is_md','is_motor','is_visual']
    rois_dict = {}
    ROI_names = ['V1','V2','V3','V3AB','hV4','IPS0','IPS1','IPS2','IPS3','LO1','LO2'];
    hemis = ['lh', 'rh'];
    n_rois = 11; n_hemis = 2;

    # initializing a dictionary to store the info about ROI definitions.
    for rk in rois_keys:
        rois_dict[rk] = [[[] for hh in range(n_hemis)] for rr in range(n_rois)]
    
    all_vox_list = []

    with h5py.File(sample_fn, 'r') as f:

        # loop over the keys into the "ROIs" element here
        # (we know in advance what they are)
        for rk in rois_keys:
            # each element is [11,2] so loop over the arrays
            for rr in range(n_rois):
                for hh in range(n_hemis):

                    # this is the confusing part...
                    # ref will be a "reference" to the element that we're looking for
                    # like <HDF5 object reference>
                    ref = f['ROIs'][rk][rr][hh]
                
                    # to get the actual data, we have to use the reference as a key into the 
                    # original dataset (f), and then convert it into np array.
                    rois_dict[rk][rr][hh] = np.array(f[ref]).astype(int)

                    if rk=='voxel_inds':
                        a = np.array(f[ref]).astype(int)
                        if len(np.shape(a))==2:
                            all_vox_list += [a]
                        # else:
                        #     # skip any that are wrong shape, empty
                        #     print('missing voxel_inds for %s %s'%(ROI_names[rr], hemis[hh]))

    # double checking that the list of voxels we pulled out is correct and lines up with all_vox_concat
    # print([av.shape for av in all_vox_list])
    all_vox_list = np.concatenate(all_vox_list, axis=0)
    assert(len(np.unique(all_vox_list))==len(all_vox_list))
    assert(np.all(np.unique(all_vox_list)==samples['all_vox_concat'][:,0]))
    
    # put the ROIs into my main dict now, for simplicity
    samples['ROIs'] = rois_dict
    samples['ROI_names'] = ROI_names
    samples['hemis'] = hemis
    
    return samples


def load_mat_behav_data(filename, varname='TheData'):
    
    """
    A way to load .mat files that contain lists of data structures
    into python as lists of dicts. This specifically works for the format of
    data used in shapedim, might need to be modified for other experiments!
    """
    
    TheData = spio.loadmat(filename, squeeze_me=True, struct_as_record=False)[varname]
    try:
        TheData = [_todict(d) for d in TheData]
    except:
        # the above will throw error if only one run
        TheData = [_todict(TheData)]
        
    return TheData

def _todict(matobj):
    '''
    A recursive function which constructs from matobjects nested dictionaries
    '''
    d = {}
    for strg in matobj._fieldnames:
        elem = matobj.__dict__[strg]
        if isinstance(elem, spio.matlab.mio5_params.mat_struct):
            d[strg] = _todict(elem)
        elif isinstance(elem, np.ndarray):
            d[strg] = _tolist(elem)
        else:
            d[strg] = elem
    return d

def _tolist(ndarray):
    '''
    A recursive function which constructs lists from cellarrays
    (which are loaded as numpy ndarrays), recursing into the elements
    if they contain matobjects.
    '''
    elem_list = []
    for sub_elem in ndarray:
        if isinstance(sub_elem, spio.matlab.mio5_params.mat_struct):
            elem_list.append(_todict(sub_elem))
        elif isinstance(sub_elem, np.ndarray):
            elem_list.append(_tolist(sub_elem))
        else:
            elem_list.append(sub_elem)
    return elem_list


def get_actual_behav_scripts():
    
    """
    From each subject's saved data file, create an .m file that contains the
    final version of the script used during their experiment. This is in 
    the field TheData(1).p.MyScript.
    """

    root = '/usr/local/serenceslab/maggie/shapeDim/'
    
    path_save = os.path.join(root,'ExpScriptsMRI','ActualVersions')
    if not os.path.exists(path_save):
        os.makedirs(path_save)

    sublist = [1,2,3,4,5,6,7,8,9,10]
    subinits = ['S%02d'%ss for ss in sublist]

    for si, ss in enumerate(sublist):

        subinit = subinits[si]
        behav_data_folder = os.path.join(root, 'DataBehavior', subinit)

        for ses in [0,1,2]:

            datadir = os.path.join(behav_data_folder,'Session%d'%(ses+1));

            # look at all data in the folder, figure out the date of the sess
            alldat = os.listdir(os.path.join(datadir))
            alldat = [d for d in alldat if '%s_MainTaskMRI_scannerversion'%subinit in d and 'TRAINING' not in d]
            sess_date = alldat[0].split('.mat')[0][-6:]
            
            filename = os.path.join(datadir, \
                                        '%s_MainTaskMRI_scannerversion_sess%d_part%d_%s.mat'%\
                                        (subinit,ses+1,1,sess_date));
            print('loading from %s'%filename)
            TheData = load_mat_behav_data(filename)
            p = TheData[0]['p']

            lines = p['MyScript'].split('\r\n')
            filename_write = os.path.join(path_save, '%s_MainTask_sess%d_%s.m'%(subinit, ses+1, sess_date))
            print('writing to %s'%filename_write)
            
            with open(filename_write, 'w') as f:
                for line in lines:
                    f.write(line + '\r\n')


def get_actual_behav_scripts_repeat_task():
    
    """
    From each subject's saved data file, create an .m file that contains the
    final version of the script used during their experiment. This is in 
    the field TheData(1).p.MyScript.
    """

    root = '/usr/local/serenceslab/maggie/shapeDim/'
    
    path_save = os.path.join(root,'ExpScriptsMRI','ActualVersions')
    if not os.path.exists(path_save):
        os.makedirs(path_save)

    sublist = [1,2,3,4,5,6,7]
    subinits = ['S%02d'%ss for ss in sublist]

    for si, ss in enumerate(sublist):

        subinit = subinits[si]
        behav_data_folder = os.path.join(root, 'DataBehavior', subinit)

        for ses in [0,1,2]:

            datadir = os.path.join(behav_data_folder,'Session%d'%(ses+1));

            # look at all data in the folder, figure out the date of the sess
            alldat = os.listdir(os.path.join(datadir))
            alldat = [d for d in alldat if '%s_OneBackTaskMRI'%subinit in d]
            sess_date = alldat[0].split('.mat')[0][-6:]
            
            if len(alldat)>0:
                filename = os.path.join(datadir, alldat[0])
                sess_date = alldat[0].split('.mat')[0][-6:]
            else:
                print('missing task for %s sess %d'%(subinit, ses+1))
                continue
                
            print('loading from %s'%filename)
            TheData = load_mat_behav_data(filename)
            p = TheData[0]['p']
            
            # print(p['MyScript'])

            lines = p['MyScript'].split('\r\n')
            filename_write = os.path.join(path_save, '%s_RepeatTask_sess%d_%s.m'%(subinit, ses+1, sess_date))
            print('writing to %s'%filename_write)
            
            with open(filename_write, 'w') as f:
                for line in lines:
                    f.write(line + '\r\n')



def get_actual_behav_scripts_silh_task():
    
    """
    From each subject's saved data file, create an .m file that contains the
    final version of the script used during their experiment. This is in 
    the field TheData(1).p.MyScript.
    """

    root = '/usr/local/serenceslab/maggie/shapeDim/'
    
    path_save = os.path.join(root,'ExpScriptsMRI','ActualVersions')
    if not os.path.exists(path_save):
        os.makedirs(path_save)

    sublist = [1,2,3,4,5,6,7]
    subinits = ['S%02d'%ss for ss in sublist]

    for si, ss in enumerate(sublist):

        subinit = subinits[si]
        behav_data_folder = os.path.join(root, 'DataBehavior', subinit)

        for ses in [0,1,2]:

            datadir = os.path.join(behav_data_folder,'Session%d'%(ses+1));

            # look at all data in the folder, figure out the date of the sess
            alldat = os.listdir(os.path.join(datadir))
            alldat = [d for d in alldat if '%s_Silhouette'%subinit in d]
            
            if len(alldat)>0:
                filename = os.path.join(datadir, alldat[0])
                sess_date = alldat[0].split('.mat')[0][-6:]
            else:
                print('missing task for %s sess %d'%(subinit, ses+1))
                continue
                
            print('loading from %s'%filename)
            try:
                TheData = load_mat_behav_data(filename)
            except:
                print('skipping %s'%subinit)
                continue
                
                
            p = TheData[0]['p']
            
            # print(p['MyScript'])

            lines = p['MyScript'].split('\r\n')
            filename_write = os.path.join(path_save, '%s_SilhouetteTask_sess%d_%s.m'%(subinit, ses+1, sess_date))
            print('writing to %s'%filename_write)
            
            with open(filename_write, 'w') as f:
                for line in lines:
                    f.write(line + '\r\n')

                    
def get_actual_behav_scripts_training_task():
    
    """
    From each subject's saved data file, create an .m file that contains the
    final version of the script used during their experiment. This is in 
    the field TheData(1).p.MyScript.
    """

    root = '/usr/local/serenceslab/maggie/shapeDim/'
    
    path_save = os.path.join(root,'ExpScriptsMRI','ActualVersions')
    if not os.path.exists(path_save):
        os.makedirs(path_save)

    sublist = [1,2,3,4,5,6,7]
    subinits = ['S%02d'%ss for ss in sublist]

    for si, ss in enumerate(sublist):

        subinit = subinits[si]
        behav_data_folder = os.path.join(root, 'DataBehavior', subinit)

        for ses in ['Training']:

            datadir = os.path.join(behav_data_folder,ses);

            # look at all data in the folder, figure out the date of the sess
            alldat = os.listdir(os.path.join(datadir))
            alldat = [d for d in alldat if 'MainTask' in d]
            
            if len(alldat)>0:
                filename = os.path.join(datadir, alldat[0])
                sess_date = alldat[0].split('.mat')[0][-6:]
            else:
                print('missing task for %s %s'%(subinit, ses))
                continue
                
            print('loading from %s'%filename)
            try:
                TheData = load_mat_behav_data(filename)
            except:
                print('skipping %s'%subinit)
                continue
                
                
            p = TheData[0]['p']
            
            # print(p['MyScript'])

            lines = p['MyScript'].split('\r\n')
            filename_write = os.path.join(path_save, '%s_MainTask_TrainingSession_%s.m'%(subinit, sess_date))
            print('writing to %s'%filename_write)
            
            with open(filename_write, 'w') as f:
                for line in lines:
                    f.write(line + '\r\n')
