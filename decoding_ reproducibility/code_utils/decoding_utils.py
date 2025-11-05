import numpy as np
import sklearn
import sklearn.svm, sklearn.discriminant_analysis

    
# helper functions for training/testing diff kinds of decoders

def decode_func_svc(X_trn, y_trn, X_tst):
    
    clf = sklearn.svm.LinearSVC()
    clf.fit(X_trn, y_trn)
    pred = clf.predict(X_tst)

    return pred

def decode_func_lda(X_trn, y_trn, X_tst):
    
    clf = sklearn.discriminant_analysis.LinearDiscriminantAnalysis()
    clf.fit(X_trn, y_trn)
    pred = clf.predict(X_tst)

    return pred

def decode_func_normeucdist(X_trn, y_trn, X_tst):
    
    pred = decoding_utils.normEucDistClass(X_trn, X_tst, y_trn)

    return pred



def normEucDistClass( train, test, group_in ):

    """
    % Calculate the NORMALIZED Euclidean distance for each trial in "test" to each of the
    % groups defined in "train"

    % IN: 
        % train is [nTrialsTraining x nVoxels];
        % test is [nTrialsTesting x nVoxels];
        % group is [nTrialsTraining x 1]

    % OUT:
        % label is [nTrialsTesting x 1] (this is the predicted label, as an
            % index into unique(group))

    % Copied from a matlab function by MMH 2/13/18
    % Adapted into python by MMH 4/5/2023
    %%
    
    """

    if len(group_in.shape)>1:
        group = np.squeeze(group_in)
    else:
        group = group_in
        
    nconds=len(np.unique(group));
    conds=np.unique(group);
    nvox = np.shape(test)[1];

    # % first, go through each condition, get its mean, variance and number of
    # % samples in training set
    meanrespeach = np.zeros((nconds,nvox));
    vareach = np.zeros((nconds,nvox));
    neach = np.zeros((nconds,1));
    for cc in range(nconds):

        # % find the trials of interest in training set    
        meanrespeach[cc,:] = np.mean(train[group==conds[cc],:],axis=0);
        vareach[cc,:] = np.var(train[group==conds[cc],:],axis=0);
        neach[cc] = np.sum(group==conds[cc]);

    # % use this to get the pooled variance for each voxel
    pooledvar = np.sum((vareach*np.tile(neach-1,[1,nvox])),axis=0, keepdims=True)/np.sum(neach-1);

    # % now loop through test set trials, and find their normalized euclidean
    # % distance to each of the training set conditions
    normEucDistAll = np.zeros((np.shape(test)[0],nconds));
    for cc in range(nconds):
        sumofsq = np.sum(((test-np.tile(meanrespeach[cc:cc+1,:],[np.shape(test)[0],1]))/ \
                          np.tile(pooledvar,[np.shape(test)[0],1]))**2,axis=1);
        normEucDistAll[:,cc] = np.sqrt(sumofsq);


    # % finally, assign a label to each of your testing set trials, choosing from
    # % the original labels in group
    colind = np.argmin(normEucDistAll, axis=1)
    # [~,colind]=min(normEucDistAll,[],2);
    label = np.zeros(np.shape(colind));
    for cc in range(nconds):
        label[colind==cc] = conds[cc]

    return label
