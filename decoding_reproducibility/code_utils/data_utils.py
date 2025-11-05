import numpy as np
import os, sys
import pandas as pd
import scipy.stats
import scipy.io as spio

from code_utils import file_utils

# root directory is 2 dirs up from this file
path = os.path.realpath(__file__).split('/')
root = '/'+os.path.join(*path[0:-3])
# root = /usr/local/serenceslab/maggie/shapeDim/

def load_main_task_labels(ss):
    
    behav_fn = os.path.join(root, 'DataBehavior', 'S%02d'%ss, \
                                          'S%02d_maintask_preproc_all.csv'%ss)
    # print('loading from %s'%behav_fn)
    bdat = pd.read_csv(behav_fn, index_col=0)

    return bdat

def load_repeat_task_labels(ss):
    
    behav_fn = os.path.join(root, 'DataBehavior', 'S%02d'%ss, \
                                          'S%02d_reptask_preproc_all.csv'%ss)
    # print('loading from %s'%behav_fn)
    bdat = pd.read_csv(behav_fn, index_col=0)

    return bdat


def load_main_task_data(ss, make_time_resolved = True, use_bigIPS = False, concat_IPS = True):
    
    """
    Load trial-by-trial data for all main task trials for a single subject (ss)
    make_time_resolved is bool for whether to return the tr-by-tr signal in 
    addition to averaged signal for each trial. 
    
    returns [main_data, main_data_by_tr, main_labels, roi_names]
    
    """
    if use_bigIPS:
        sample_fn = os.path.join(root, 'Samples','SampleFile_bigIPS_S%02d.mat'%ss)
    else:
        sample_fn = os.path.join(root, 'Samples','SampleFile_S%02d.mat'%ss)

    # print('loading from %s'%sample_fn)

    samples = file_utils.load_samplefile_h5py(sample_fn)

    samples['all_vox_concat'].shape

    import scipy.io as spio

    # load the "timing" file (what happened on each TR)
    # this is made in GetEventTiming.m
    timing_fn = os.path.join(root, 'Samples','TimingFile_S%02d.mat'%ss)
    # print('loading from %s'%timing_fn)
    timing = spio.loadmat(timing_fn, squeeze_me=True, struct_as_record=False)
    main = file_utils._todict(timing['main'])
    rep = file_utils._todict(timing['rep'])

    # going to check labels from the "timing" file against my "behav" file
    # the .csv file was made in process_behav.py
    # the labels in here should match what is in timing file
    # will compare them here to verify that we get the exact same sequence.
    behav_fn = os.path.join(root, 'DataBehavior', 'S%02d'%ss, \
                                          'S%02d_maintask_preproc_all.csv'%ss)
    # print('loading from %s'%behav_fn)
    bdat = pd.read_csv(behav_fn, index_col=0)

    
    if ss<=7:
        nTRs = 327-16;
        # which TRs am i averaging over? From the target onset time.
        avgTRs_targ = [4,7];
        # TRs are 0.8 seconds in length. So this is 3.2 - 5.6 seconds after onset.
        # TRs go like...TR 0 = 0 seconds, TR 1 = 0.8, seconds, TR 2 = 1.6 seconds, and so on.
        
        # if we're returning time-resolved data, how many TRs to include per trial?
        nTRs_concat = 14;
        
    else:
        # Prisma setup
        nTRs = 201;
        # adjusting the above numbers to account for different TR length in sec.
        # the TR is 1.3 seconds with Prisma, so to get the same time window in seconds, we 
        # want to grab earlier TRs. This is approximate but should be close.
        # Taking average over a window 4 TRs long, as we did above.
        avgTRs_targ = [2,5]
        
        # if we're returning time-resolved data, how many TRs to include per trial?
        # again this has to be smaller here
        nTRs_concat = 9;
        
    
    nRunsExpected = 36;
    nTrialsPerRun = 48;

    roi_names = samples['ROI_names']
    
    n_rois = len(roi_names)
    n_hemis = len(samples['hemis'])

    main_data = []
    main_data_by_tr = []

    for rr in range(n_rois):

        dat_this_roi = []

        for hh in range(n_hemis):

            roi_inds_num = np.array(samples['ROIs']['voxel_inds'][rr][hh])

            inds_this_hemi = np.isin(np.array(samples['all_vox_concat']), roi_inds_num)[:,0]

            if np.sum(inds_this_hemi)>0:
                # print(np.sum(inds_this_hemi))
                # samples['samplesMain'] is originally [nVox x nTRs]
                # transpose because i want [nTRs x nVox]
                dat_this_hemi = samples['samplesMain'][inds_this_hemi,:].T
                dat_this_roi += [dat_this_hemi]
            # else:
            #     print('missing data for %s %s'%(roi_names[rr], samples['hemis'][hh]))

        dat_this_roi = np.concatenate(dat_this_roi, axis=1)
        # print(dat_this_roi.shape)
        nVox = dat_this_roi.shape[1]
        # print('processing area %s, %d voxels'%(roi_names[rr],nVox))

        # count things, check the counts
        nTRsTotal = dat_this_roi.shape[0]
        assert(np.mod(nTRsTotal, nTRs)==0)
        nRuns = int(nTRsTotal / nTRs)
        assert(nTRsTotal==len(main['RunLabels']))
        assert(nRuns==len(np.unique(main['RunLabels'])))
        # if (rr==0) and (nRuns!=nRunsExpected):
        #     print('warning: you only have %d/%d runs for S%02d\n'%(nRuns, nRunsExpected, ss))

        # label the onset of each trial
        # 1 = stim on, 0 = stim off
        event_diff  = np.diff(np.array([0] + main['EventLabels']))
        event_diff_reshaped = np.reshape(event_diff, [nTRs, nRuns], order='F')
        trial_onset_bool = event_diff_reshaped==1;
        trial_onset_bool = np.reshape(trial_onset_bool, [nTRs*nRuns,1], order='F')
        trial_onset_num = np.where(trial_onset_bool)[0]

        nTrials = nRuns*nTrialsPerRun
        assert(len(trial_onset_num)==nTrials)

        if rr==0:

            # now checking the trial labels

            # check the recorded responses are identical between these sequences
            resp1 = np.array(main['ResponseLabels'])[trial_onset_num].astype(float)
            resp1[resp1==0] = np.nan
            resp2 = np.array(bdat['resp'])
            has_nans = np.isnan(resp1)
            assert(np.all(np.isnan(resp2[has_nans])))
            assert(np.all(resp1[~has_nans]==resp2[~has_nans]))

            # check a couple more variables...make sure they line up
            vals1 = np.array(main['IsMainLabels'])[trial_onset_num]
            vals2 = np.array(bdat['is_main_grid'])
            assert(np.all(vals1==vals2))

            vals1 = np.array(main['PointLabels'])[:,0][trial_onset_num]
            vals2 = np.array(bdat['ptx'])
            assert(np.all(vals1==vals2))

            vals1 = np.array(main['PointLabels'])[:,1][trial_onset_num]
            vals2 = np.array(bdat['pty'])
            assert(np.all(vals1==vals2))

            vals1 = np.array(main['BoundLabels'])[trial_onset_num]
            vals2 = np.array(bdat['task'])
            assert(np.all(vals1==vals2))

            vals1 = np.array(main['MapLabels'])[trial_onset_num]
            vals2 = np.array(bdat['map'])
            assert(np.all(vals1==vals2))

            vals1 = np.array(main['SessLabels'])[trial_onset_num]
            vals2 = np.array(bdat['sess'])
            assert(np.all(vals1==vals2))

            vals1 = np.array(main['RunLabels'])[trial_onset_num]
            vals2 = np.array(bdat['run_overall'])
            assert(np.all(vals1==vals2))
            
            vals1 = np.array(main['TrialLabels'])[trial_onset_num]
            vals2 = np.array(bdat['trial_overall'])
            assert(np.all(vals1==vals2))

        # zscore the data from each run to normalize.
        for run in np.unique(main['RunLabels']):

            run_inds = main['RunLabels']==run
            assert(np.sum(run_inds)==nTRs)

            dat_this_roi[run_inds,:] = scipy.stats.zscore(dat_this_roi[run_inds,:], axis=0)

        # gather data for each individual trial.
        # average over a fixed time window following stim onset.
        dat_avg_targ = np.full(fill_value=np.nan, shape=(nTrials, nVox))

        if make_time_resolved:
            dat_by_tr = np.full(fill_value=np.nan, shape=(nTrials, nTRs_concat, nVox))
        else:
            dat_by_tr = None
            
        triCnt = -1
        # counter across "good" trials where the entire desired avg window is available.
        # should be all of them

        # looping over runs, this ensures that we never accidentally go beyond end of
        # a single run 
        for run in np.unique(main['RunLabels']):

            run_inds = main['RunLabels']==run
            assert(np.sum(run_inds)==nTRs)

            cur_dat = dat_this_roi[run_inds,:]

            # find onsets of trials, within this run only
            these_targ_onsets = np.where(trial_onset_bool[run_inds])[0]

            assert(len(these_targ_onsets)==nTrialsPerRun)

            for tt in these_targ_onsets:

                window_start = tt + avgTRs_targ[0]
                window_stop = tt + avgTRs_targ[1]

                if window_stop<nTRs:

                    triCnt+=1

                    # average the data over desired interval.
                    dat_avg_targ[triCnt,:] = np.mean(cur_dat[window_start:window_stop, :], axis=0)
                
                
                if make_time_resolved:
                    
                    for tr in range(nTRs_concat):
            
                        # take the specified timepoint and put it into my big array
                        dat_by_tr[triCnt, tr, :] = cur_dat[tt+tr, :]
                        

        assert(triCnt==nTrials-1)
        assert(not np.any(np.isnan(dat_avg_targ.ravel())))
        if make_time_resolved:
            assert(not np.any(np.isnan(dat_by_tr.ravel())))


        main_data += [dat_avg_targ]    
        
        main_data_by_tr += [dat_by_tr]
        

    if concat_IPS:

        # going to combine all the 4 ips subregions together
        # attempting to boost signal since these areas are small in some subjects.
        # the rois will now go:
        # ['V1','V2','V3','V3AB','hV4','LO1','LO2','IPSall']
        ips_roi = [5,6,7,8]

        roi_names = roi_names[0:5] + roi_names[9:] + ['IPSall']
        n_rois = len(roi_names)

        main_data_new = main_data[0:5] + main_data[9:]
        ips_concat = np.concatenate([main_data[rr] for rr in ips_roi], axis=1)
        main_data_new += [ips_concat]  
        main_data = main_data_new

        if make_time_resolved:
            main_data_by_tr_new = main_data_by_tr[0:5] + main_data_by_tr[9:]
            ips_concat = np.concatenate([main_data_by_tr[rr] for rr in ips_roi], axis=2)
            main_data_by_tr_new += [ips_concat]  
            main_data_by_tr = main_data_by_tr_new

    main_labels = bdat; # these will be trial-wise labels for each trial

    
    return main_data, main_data_by_tr, main_labels, roi_names



def load_repeat_task_data(ss, make_time_resolved = True,  use_bigIPS = False, concat_IPS = True):

    """
    Load trial-by-trial data for all repeat (one-back) task trials 
    for a single subject (ss)

    make_time_resolved is bool for whether to return the tr-by-tr signal in 
    addition to averaged signal for each trial. 

    returns [rep_data, rep_data_by_tr, rep_labels, roi_names]

    """
    if use_bigIPS:
        sample_fn = os.path.join(root, 'Samples','SampleFile_bigIPS_S%02d.mat'%ss)
    else:
        sample_fn = os.path.join(root, 'Samples','SampleFile_S%02d.mat'%ss)

    # print('loading from %s'%sample_fn)

    samples = file_utils.load_samplefile_h5py(sample_fn)

    samples['all_vox_concat'].shape

    import scipy.io as spio

    # load the "timing" file (what happened on each TR)
    # this is made in GetEventTiming.m
    timing_fn = os.path.join(root, 'Samples','TimingFile_S%02d.mat'%ss)
    # print('loading from %s'%timing_fn)
    timing = spio.loadmat(timing_fn, squeeze_me=True, struct_as_record=False)
    main = file_utils._todict(timing['main'])
    rep = file_utils._todict(timing['rep'])

    # going to check labels from the "timing" file against my "behav" file
    # the .csv file was made in process_behav.py
    # the labels in here should match what is in timing file
    # will compare them here to verify that we get the exact same sequence.
    behav_fn = os.path.join(root, 'DataBehavior', 'S%02d'%ss, \
                                          'S%02d_reptask_preproc_all.csv'%ss)
    # print('loading from %s'%behav_fn)
    bdat = pd.read_csv(behav_fn, index_col=0)

    
    if ss<=7:
        nTRs = 329-16;
        # which TRs am i averaging over? From the target onset time.
        avgTRs_targ = [4,7];
        # TRs are 0.8 seconds in length. So this is 3.2 - 5.6 seconds after onset.
        # TRs go like...TR 0 = 0 seconds, TR 1 = 0.8, seconds, TR 2 = 1.6 seconds, and so on.
        
        # if we're returning time-resolved data, how many TRs to include per trial?
        nTRs_concat = 14;
    
    else:
        # Prisma setup
        nTRs = 203;
        # adjusting the above numbers to account for different TR length in sec.
        # the TR is 1.3 seconds with Prisma, so to get the same time window in seconds, we 
        # want to grab earlier TRs. This is approximate but should be close.
        # Taking average over a window 4 TRs long, as we did above.
        avgTRs_targ = [2,5]
        
        # if we're returning time-resolved data, how many TRs to include per trial?
        # again this has to be smaller here
        nTRs_concat = 9;
        
    nRunsExpected = 12;
    nTrialsPerRun = 48;

    roi_names = samples['ROI_names']
    n_rois = len(roi_names)
    n_hemis = len(samples['hemis'])

    rep_data = []
    rep_data_by_tr = []

    for rr in range(n_rois):

        dat_this_roi = []

        for hh in range(n_hemis):

            roi_inds_num = np.array(samples['ROIs']['voxel_inds'][rr][hh])

            inds_this_hemi = np.isin(np.array(samples['all_vox_concat']), roi_inds_num)[:,0]

            if np.sum(inds_this_hemi)>0:
                # print(np.sum(inds_this_hemi))
                # samples['samplesRep'] is originally [nVox x nTRs]
                # transpose because i want [nTRs x nVox]
                dat_this_hemi = samples['samplesRep'][inds_this_hemi,:].T
                dat_this_roi += [dat_this_hemi]
            # else:
            #     print('missing data for %s %s'%(roi_names[rr], samples['hemis'][hh]))

        dat_this_roi = np.concatenate(dat_this_roi, axis=1)
        # print(dat_this_roi.shape)
        nVox = dat_this_roi.shape[1]
        # print('processing area %s, %d voxels'%(roi_names[rr],nVox))

        # count things, check the counts
        nTRsTotal = dat_this_roi.shape[0]
        assert(np.mod(nTRsTotal, nTRs)==0)
        nRuns = int(nTRsTotal / nTRs)
        assert(nTRsTotal==len(rep['RunLabels']))
        assert(nRuns==len(np.unique(rep['RunLabels'])))
        # if (rr==0) and (nRuns!=nRunsExpected):
        #     print('warning: you only have %d/%d runs for S%02d\n'%(nRuns, nRunsExpected, ss))

        # label the onset of each trial
        # 1 = stim on, 0 = stim off
        event_diff  = np.diff(np.array([0] + rep['EventLabels']))
        event_diff_reshaped = np.reshape(event_diff, [nTRs, nRuns], order='F')
        trial_onset_bool = event_diff_reshaped==1;
        trial_onset_bool = np.reshape(trial_onset_bool, [nTRs*nRuns,1], order='F')
        trial_onset_num = np.where(trial_onset_bool)[0]

        nTrials = nRuns*nTrialsPerRun
        assert(len(trial_onset_num)==nTrials)

        if rr==0:

            # now checking the trial labels

            # check the recorded responses are identical between these sequences
            resp1 = np.array(rep['ResponseLabels'])[trial_onset_num].astype(float)
            resp1[resp1==0] = np.nan
            resp2 = np.array(bdat['resp'])
            has_nans = np.isnan(resp1)
            assert(np.all(np.isnan(resp2[has_nans])))
            assert(np.all(resp1[~has_nans]==resp2[~has_nans]))

            # check a couple more variables...make sure they line up
            vals1 = np.array(rep['IsMainLabels'])[trial_onset_num]
            vals2 = np.array(bdat['is_main_grid'])
            assert(np.all(vals1==vals2))

            vals1 = np.array(rep['PointLabels'])[:,0][trial_onset_num]
            vals2 = np.array(bdat['ptx'])
            assert(np.all(vals1==vals2))

            vals1 = np.array(rep['PointLabels'])[:,1][trial_onset_num]
            vals2 = np.array(bdat['pty'])
            assert(np.all(vals1==vals2))

            vals1 = np.array(rep['MapLabels'])[trial_onset_num]
            vals2 = np.array(bdat['map'])
            assert(np.all(vals1==vals2))

            vals1 = np.array(rep['SessLabels'])[trial_onset_num]
            vals2 = np.array(bdat['sess'])
            assert(np.all(vals1==vals2))

            vals1 = np.array(rep['RunLabels'])[trial_onset_num]
            vals2 = np.array(bdat['run_overall'])
            assert(np.all(vals1==vals2))

            vals1 = np.array(rep['TrialLabels'])[trial_onset_num]
            vals2 = np.array(bdat['trial_overall'])
            assert(np.all(vals1==vals2))

        # zscore the data from each run to normalize.
        for run in np.unique(rep['RunLabels']):

            run_inds = rep['RunLabels']==run
            assert(np.sum(run_inds)==nTRs)

            dat_this_roi[run_inds,:] = scipy.stats.zscore(dat_this_roi[run_inds,:], axis=0)

        # gather data for each individual trial.
        # average over a fixed time window following stim onset.
        dat_avg_targ = np.full(fill_value=np.nan, shape=(nTrials, nVox))

        if make_time_resolved:
            dat_by_tr = np.full(fill_value=np.nan, shape=(nTrials, nTRs_concat, nVox))
        else:
            dat_by_tr = None

        triCnt = -1
        # counter across "good" trials where the entire desired avg window is available.
        # should be all of them

        # looping over runs, this ensures that we never accidentally go beyond end of
        # a single run 
        for run in np.unique(rep['RunLabels']):

            run_inds = rep['RunLabels']==run
            assert(np.sum(run_inds)==nTRs)

            cur_dat = dat_this_roi[run_inds,:]

            # find onsets of trials, within this run only
            these_targ_onsets = np.where(trial_onset_bool[run_inds])[0]

            assert(len(these_targ_onsets)==nTrialsPerRun)

            for tt in these_targ_onsets:

                window_start = tt + avgTRs_targ[0]
                window_stop = tt + avgTRs_targ[1]

                if window_stop<nTRs:

                    triCnt+=1

                    # average the data over desired interval.
                    dat_avg_targ[triCnt,:] = np.mean(cur_dat[window_start:window_stop, :], axis=0)


                if make_time_resolved:

                    for tr in range(nTRs_concat):

                        # take the specified timepoint and put it into my big array
                        dat_by_tr[triCnt, tr, :] = cur_dat[tt+tr, :]


        assert(triCnt==nTrials-1)
        assert(not np.any(np.isnan(dat_avg_targ.ravel())))
        if make_time_resolved:
            assert(not np.any(np.isnan(dat_by_tr.ravel())))


        rep_data += [dat_avg_targ]    

        rep_data_by_tr += [dat_by_tr]


        
    if concat_IPS:

        # going to combine all the 4 ips subregions together
        # attempting to boost signal since these areas are small in some subjects.
        # the rois will now go:
        # ['V1','V2','V3','V3AB','hV4','LO1','LO2','IPSall']
        ips_roi = [5,6,7,8]

        roi_names = roi_names[0:5] + roi_names[9:] + ['IPSall']
        n_rois = len(roi_names)

        rep_data_new = rep_data[0:5] + rep_data[9:]
        ips_concat = np.concatenate([rep_data[rr] for rr in ips_roi], axis=1)
        rep_data_new += [ips_concat]  
        rep_data = rep_data_new

        if make_time_resolved:
            rep_data_by_tr_new = rep_data_by_tr[0:5] + rep_data_by_tr[9:]
            ips_concat = np.concatenate([rep_data_by_tr[rr] for rr in ips_roi], axis=2)
            rep_data_by_tr_new += [ips_concat]  
            rep_data_by_tr = rep_data_by_tr_new

            
    rep_labels = bdat; # these will be trial-wise labels for each trial


    return rep_data, rep_data_by_tr, rep_labels, roi_names