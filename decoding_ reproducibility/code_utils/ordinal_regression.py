import numpy as np
import sklearn
import sklearn.svm, sklearn.discriminant_analysis, sklearn.linear_model
import time

def get_model(C=0.01, n_threads=8):
    
    # creating the classifier that goes into the ordinal regression algorithm
    model = sklearn.linear_model.LogisticRegression(C = C, \
                                                    solver='lbfgs', \
                                                    penalty='l2', \
                                                    n_jobs = n_threads , \
                                                    max_iter = 1000, 
                                                    class_weight = 'balanced')
    # Note: it seems like we need to keep C fixed for all the individual
    # classifiers in the full ordinal regression algorithm.
    # Otherwise it can give bad results especially for middle values.
    
    return model


class ordinal_regress_model():
    
    def __init__(self, clf_func):
        
        # clf_func is a function that will generate a classifier object
        # (in this case sklearn.linear_model.LogisticRegression)
        # can pass more arguments in during fitting (model_pars)
        self.clf_func = clf_func
        
    def fit(self, X, y, model_pars=[]):
        
        # y values will be sorted lowest to highest
        self.unique_y = np.unique(y)
        self.n_y = len(self.unique_y)
        self.n_clf = self.n_y - 1

        # make sets of binary labels for our different classifiers
        ylabs_binary = [(y > y0).astype(int) for y0 in self.unique_y[0:-1]]
        
        self.models_fitted = []

        for yi in range(self.n_clf):

            # create the classifier object (can pass in arguments here)
            model = self.clf_func(*model_pars)

            model.fit(X, ylabs_binary[yi])

            self.models_fitted += [model]
            

    def predict(self, X):

        # compute prediction for each individual classifier
        # these values will represent probability of >value
        preds_binary = np.array([self.models_fitted[yi].predict_proba(X)[:,1] \
                                 for yi in range(self.n_clf)]).T
    
        # now convert these back into probabilities for individual classes
        prob_each = np.zeros((X.shape[0], self.n_y), dtype=np.float64)
        
        for yi in range(self.n_y):
            if yi==0:
                # smallest value, classifier is >this value, so take 1-prob
                prob_each[:,yi] = 1-preds_binary[:,0]
            elif yi==(self.n_y-1):
                # biggest value, classifier is >previous value
                prob_each[:,yi] = preds_binary[:,yi-1]
            else:
                # middle value, classifier is >previous value - >this value
                prob_each[:,yi] = preds_binary[:,yi-1] - preds_binary[:,yi] 

        # prob of all classes should sum up to 1
        assert(np.all(np.sum(prob_each, axis=1).round(9)==1))

        # get max prob as predicted label
        lab_ind = np.argmax(prob_each, axis=1)
        lab = self.unique_y[lab_ind]
        
        return lab, prob_each