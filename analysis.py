#! /usr/bin/env python

"""
Copyright of the program:
    Andrea Agazzi, UNIGE and Stanford University
    Vincent Deo, Stanford University

This script performs a cross-validation on a series of pipelines for the analysis of metabolomics data for the identification of early-stage biomarkers for type II diabetes mellitus

"""

import numpy as np
import logging
import src.importd
import cv_sampling as cv
from sklearn import *
import sklearn as sk
from sklearn.pipeline import Pipeline
from sklearn.grid_search import GridSearchCV
from datetime import datetime
import time
import pipe2 as pipeline
import sys


if __name__ == '__main__':

    if 'p' in sys.argv[1::]:
        files = [('data/file_plasma.dat','data/file3_plasma.dat')]
    elif 'a' in sys.argv[1::]:
        files = [('data/file.dat','data/file3.dat'),('data/file_plasma.dat','data/file3_plasma.dat')]
    else:
        [('data/file.dat','data/file2.dat')]
    print files
    # initialize X and Y for tests
    for (file1,file2) in files:
        wids = ['week_4','week_5','week_6','week_10','week4','week5','week6','week10']
        ns, Xdata,Ydata = src.importd.importfile(file1)
        ns, X, Y, _ = src.importd.filterd(ns,Xdata,Ydata,wids)
        _, names = src.importd.import_cnames(file2)
        # run automated tests
        pipelst = [['FFS','L1LogReg'],['PCA','L1LogReg'],['FFS','FDA'],['PCA','FDA'],['FFS','RF'],['PCA','RF']]
        if len(sys.argv) != 1 and 's' in sys.argv[1::]:
            pardict = dict(FDA__store_covariance=[True],FFS__k=range(10,211,50),RF__n_estimators=range(10,311,100),PCA__whiten=[True,False],PCA__n_components=range(10,151,20),L1LogReg__C=list(np.logspace(-9,0,10)))
        else:
            pardict = dict(FDA__store_covariance=[True],FFS__k=range(10,201,5),RF__n_estimators=range(10,301,30),PCA__whiten=[True,False],PCA__n_components=range(10,201,5),L1LogReg__C=list(np.logspace(-9,0)))

        for pipeel in pipelst:
            #run an initialization test for a pipeline with pca and fda
            pipe = pipeline.Pipe(X,Y,names,wids)
            pipe.setpipe(pipeel)

            # cvcounter test
            print('Pipe.cvcounter =\t'+str(pipe.cvcounter))

            print(np.shape(np.array(X)))

            griddic = dict()
            for p in pardict.keys():
                if True in [True if q in p else False for q in pipeel]:
                    griddic[p] = pardict[p]

            pipe.crossgrid(griddic,crossval=cv.leave_x_out(pipe.Y, 10, nsamples=200, testlst=[i for i,n in enumerate(ns) if ('week_4' in n or 'week_5' in n or 'week4' in n or 'week5' in n)]))
            pipe.return_score()
            pipe.return_rank()
            pipe.return_ranks(.9,printtofile=True)
