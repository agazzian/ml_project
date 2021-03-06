#! /usr/bin/env python

"""
Copyright of the program: Andrea Agazzi, UNIGE


"""

import numpy as np
import logging
import jmport
import cv_sampling as cv
from sklearn import *
import sklearn as sk
from sklearn.pipeline import Pipeline
from sklearn.grid_search import GridSearchCV
from datetime import datetime
import time

class Pipe(object):
    """
    pipeline class
    """
    cvcounter = 0

    def __init__(self,Xdata,Ydata,feat_names,weeks,pipe=None,griddic=None,cv=None,organ='liver',isTargeted=False,LogConc=False):
        self.X = Xdata
        self.Y = Ydata
        self.feat_names = feat_names
        self.weeks = weeks
        self._pipe = pipe
        self._pipelst = []
        self._bestscore = None
        self.grid = griddic
        self._gridsearch = None
        self._biolst = [None for n in feat_names]
        self.cv = cv
        self.organ = organ
        self.isTargeted = isTargeted
        self.LogConc = LogConc

    def newpipe(self):
        """ if pipe is empty return true else reset all """
        pass

    def setpipe(self, nlst):
        """ set a new pipe attribute """
        steplst = []
        # check if there are no doubles in the pipeline or more in general if the pipeline is valid
        for n in nlst:
            steplst.append(Pipe.addpipeel(n))
            print(n+' added to the pipeline')

        self._pipe = Pipeline(steps=steplst)
        self._pipelst = nlst[:]
        print(self._pipelst)

    @staticmethod
    def addpipeel(estimatorname):
        """
        Associates the names in the list [PCA, thresh, logreg, FDA, RF] a sklearn estimator.

        Returns a tuple (sk.estimator, estimatorname) where
            sk.estimator is the sklearn estimator object corresponding to the input string
            estimatorname is the input string
        In case the input string is not in the list of accepted inputs the method
        returns an error message and None
        """

        estimatorSwitcher = {
            'PCA': sk.decomposition.PCA(copy=True),
            'FFS': sk.feature_selection.SelectKBest(score_func=corr_analysis),
            'L1LogReg': sk.linear_model.LogisticRegression(penalty='l1'),
            'L2LogReg': sk.linear_model.LogisticRegression(),
            'FDA': sk.discriminant_analysis.LinearDiscriminantAnalysis(n_components = 1,solver='svd'),
            'RF': sk.ensemble.RandomForestClassifier(),
            'GB': sk.ensemble.GradientBoostingClassifier()
        }

        try:
            result = (estimatorname, estimatorSwitcher[estimatorname])
        except KeyError:
            print('Error: estimator '+estimatorname+' not in the list!\n')
            result = None

        return result

    def crossgrid(self, griddic, cv=None):
        """
        perform a crossvalidation procedure for mparameters in grid and cv-sample cv

        inputs
            grid: dictionary of the form
                dict(ESTIMATORNAME1__PARAMETER1 = PARAMLST1,
                     ESTIMATORNAME2__PARAMETER2 = PARAMLST2,
                     ...
                     )
            cv: crossvalidation array of the form [IDn1, IDn2,...IDnN]
        """
        # initialize cv procedure
        if cv != None:
            # if cv not empty overwrite existing cv procedure
            self.cv = cv
        elif cv == None and self.cv == None:
            # if no cv procedure has been specified set the classical l20o
            print('ATTENTION:\tNo CV procedure specified, proceeding with reduced l20o.')
            cv = cv.leave_x_out(self.Y,20)

        # initialize the _gridsearch attribute
        # need to include how to create the dictionary from the input
        self._gridsearch = sk.grid_search.GridSearchCV(self._pipe, griddic, n_jobs=-1) # @UndefinedVariable

        # fit the CV grid
        self._gridsearch.fit(self.X,self.Y)

    def return_score(self):
        """ returns the best score of the fitted model """
        self._bestscore = self._gridsearch.best_score_
        return self._bestscore

    @staticmethod
    def coeffs(estimator):
        """
        Extracts the coefficients associated to each metabolite

        Works only for supervised algorithms
        """
        coeflst = []
        if hasattr(estimator,'feature_importances_'):
            # RandomForestClassifier
            coeflst = estimator.feature_importances_
        elif hasattr(estimator,'coef_'):
            # FDA or LogReg
            coeflst = [abs(c) for c in estimator.coef_]
        elif hasattr(estimator,'scores_') and hasattr(estimator,'get_support'):
            # Greedy correlation filter
            supp = estimator.get_support()
            coeflst = [abs(c) if supp[i] else 0 for i,c in enumerate(estimator.scores_)]
        return coeflst

    def return_rank(self,*args):
        """ returns the biomarker ranking of the best performing fitting algorithm """
        biolst = [None for i in self.feat_names]
        if len(args) == 0:
            estimator = self._gridsearch.best_estimator_
            self._biolst = biolst
        else:
            estimator = args[0]

        for stepname in self._pipelst:
            clst = Pipe.coeffs(estimator.named_steps[stepname])
            if stepname == 'FDA':
                clst = list(clst[0])
            # add the scores to the biolst dictionary if they are not 0
            print(clst)
            counter = 0
            for i,c in enumerate(biolst):
                if c != 0:
                    biolst[i] = clst[counter]
                    counter += 1

        return biolst

    def return_ranks(self,tol,printtofile=False):
        """ returns the averaged biomarker rankings for all fits that perform within (tol*bestscore, bestscore)"""

        ranks = []

        for score_triple in self._gridsearch.grid_scores_[:]:
            # if the score is above the tolerance refit the model and return the ranking
            if score_triple[1] >= tol*self._bestscore:
                steplst = []
                # check if there are no doubles in the pipeline or more in general if the pipeline is valid
                for n in self._pipelst[:]:
                    steplst.append(Pipe.addpipeel(n))

                pipetofit = Pipeline(steps=steplst)

                print(score_triple[0])
                estimator = pipetofit.set_params(**score_triple[0]).fit(self.X,self.Y)
                ranks.append((score_triple[1],score_triple[0],pipe.return_rank(estimator)))

        if printtofile == True:
            self.save_ranks(ranks)

        return ranks

    def save_ranks(self,ranks):
        """
        save the ranks list in a file

        The saved rank list is sorted according to the score
        """
        now = datetime.now()
        with open('../results/ranks/'+self.organ+'_Targ='+str(self.isTargeted)+'_Log='+str(self.LogConc)+'_'+'_'.join(self._pipelst)+'_'+now.strftime('%Y_%m_%d_%H_%M')+'_'+str(Pipe.cvcounter)+'.dat','w') as f:
            f.write('# weeks: \t'+','.join(self.weeks)+'\n# pipeline:\t'+'+'.join(self._pipelst)+'\n cv:\tleave-'+str(len(list(self.cv[0][1])))+'-out \t samples: \t'+str(len(list(self.cv)))+'\n\n')
            for l in sorted(ranks,key= lambda x: x[0],reverse=True):
                f.write('score:\t'+str(l[0])+'\nparameters:\t'+str(l[1])+'\n'+'\n'.join([a+'\t'+b for (a,b) in sorted(zip(map(str,l[2]),self.feat_names),key = lambda x: x[0],reverse=True)])+'\n\n------------------------------------------------\n')


def corr_analysis(Xdata,Ydata):
    """
	Pearson correlation analysis fo the features in xdata and ydata
	"""
    return np.array([abs(np.corrcoef(np.array(Xdata)[:,j],Ydata)[0][1]) for j in range(len(Xdata[0]))]),np.array([1 for j in range(len(Xdata[0]))])

if __name__ == '__main__':

    # initialize X and Y for tests

    organ = 'liver'
    isTargeted = False #if none use both
    LogConc = True
    # this need to be changed depending on plasma/liver ---------------------------------------------------------------------------------------------------------------
    if organ == 'liver' and (isTargeted == False or isTargeted == None):
        wids = ['week4','week5','week6','week10']
    elif organ == 'plasma' and isTargeted == False:
        wids = ['week4','week5','week6','week10']
    elif organ == 'liver' and isTargeted == True:
        wids = ['4w','5w','6w','10w']
    cvweeks = wids[0:2]

    pns,cns,Xdata,Ydata = jmport.importdata(organ,isTargeted=isTargeted,LogConc=LogConc)
    _, X, Y, _ = jmport.filterd(pns,Xdata,Ydata,wids)

    # run automated tests

    #run an initialization test for a pipeline with pca and fda
    pipe = Pipe(X,Y,cns,wids,organ=organ,isTargeted=isTargeted,LogConc=LogConc)
    pipe.setpipe(['FFS','GB'])

    # cvcounter test
    print('Pipe.cvcounter =\t'+str(pipe.cvcounter))

    print(np.shape(np.array(X)))
    # test initialization of grid parameters


    #griddic = dict(FFS__k=[10,20,40,50,100,130,200,750,800],RF__n_estimators=[10,100,200,300],RF__criterion=["gini","entropy"],RF__max_features=["sqrt","log2"])
    griddic = dict(FFS__k=[10,20,40,50,100,130,200,750,800],GB__n_estimators=[100,200,300,600],GB__learning_rate=[0.1,0.3,0.5],GB__max_features=["auto"],GB__max_depth=[3,5,10])
    pipe.crossgrid(griddic,cv=cv.leave_x_out(pipe.Y,20,nsamples=200,testlst=[i for i,n in enumerate(pns) if any(j in n for j in cvweeks)]))

    print(pipe.return_score())
    print(pipe._gridsearch.grid_scores_)
    print(pipe._pipe.named_steps.keys())
    pipe.return_rank()
    pipe.return_ranks(.9,printtofile=True)
