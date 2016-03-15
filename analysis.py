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


if __name__ == '__main__':

    # initialize X and Y for tests

    wids = ['week_4','week_5','week_6','week_10']

    ns, Xdata,Ydata = src.importd.importfile('../data/file.dat')
    ns, X, Y, _ = src.importd.filterd(ns,Xdata,Ydata,wids)
    _, names = src.importd.import_cnames('../data/file3.dat')

    # run automated tests

    #run an initialization test for a pipeline with pca and fda
    pipe = pipeline.Pipe(X,Y,names,wids)
    pipe.setpipe(['FFS','RF'])

    # cvcounter test
    print('Pipe.cvcounter =\t'+str(pipe.cvcounter))

    print(np.shape(np.array(X)))
    # test initialization of grid parameters

    griddic = dict(FFS__k=range(10,501,5),RF__n_estimators=range(10,401,10))
    pipe.crossgrid(griddic,cv=cv.leave_x_out(pipe.Y, 10, nsamples=200, testlst=[i for i,n in enumerate(ns) if ('4' in n or '5' in n)]))

##############################################################

    pipe.return_ranks(.9,printtofile=True)

    pipe = pipeline.Pipe(X,Y,names,wids)
    pipe.setpipe(['FFS','FDA'])

    # cvcounter test
    # test initialization of grid parameters

    griddic = dict(FFS__k=range(10,501,5))
    pipe.crossgrid(griddic,cv=cv.leave_x_out(pipe.Y, 10, nsamples=200, testlst=[i for i,n in enumerate(ns) if ('4' in n or '5' in n)]))

    pipe.return_rank()
    pipe.return_ranks(.9,printtofile=True)

###############################################################

    pipe = pipeline.Pipe(X,Y,names,wids)
    pipe.setpipe(['FFS','LogReg'])

    # cvcounter test
    # test initialization of grid parameters

    griddic = dict(FFS__k=range(10,501,5),LogReg__C=list(np.logspace(-9,0)))
    pipe.crossgrid(griddic,cv=cv.leave_x_out(pipe.Y, 10, nsamples=200, testlst=[i for i,n in enumerate(ns) if ('4' in n or '5' in n)]))

    pipe.return_rank()
    pipe.return_ranks(.9,printtofile=True)

###############################################################

    pipe = pipeline.Pipe(X,Y,names,wids)
    pipe.setpipe(['PCA','FDA'])

    # cvcounter test
    # test initialization of grid parameters

    griddic = dict(PCA__n_components=range(10,501,5))
    pipe.crossgrid(griddic,cv=cv.leave_x_out(pipe.Y, 10, nsamples=200, testlst=[i for i,n in enumerate(ns) if ('4' in n or '5' in n)]))

    pipe.return_rank()
    pipe.return_ranks(.9,printtofile=True)

###############################################################

    pipe = pipeline.Pipe(X,Y,names,wids)
    pipe.setpipe(['PCA','LogReg'])

    # cvcounter test
    # test initialization of grid parameters

    griddic = dict(PCA__n_components=range(10,501,5),LogReg__C=list(np.logspace(-9,0)))
    pipe.crossgrid(griddic,cv=cv.leave_x_out(pipe.Y, 10, nsamples=200, testlst=[i for i,n in enumerate(ns) if ('4' in n or '5' in n)]))

    pipe.return_rank()
    pipe.return_ranks(.9,printtofile=True)

###############################################################

    pipe = pipeline.Pipe(X,Y,names,wids)
    pipe.setpipe(['PCA','RF'])

    # cvcounter test
    # test initialization of grid parameters

    griddic = dict(PCA__n_components=range(10,501,5),RF__n_estimators=range(10,401,10))
    pipe.crossgrid(griddic,cv=cv.leave_x_out(pipe.Y, 10, nsamples=200, testlst=[i for i,n in enumerate(ns) if ('4' in n or '5' in n)]))

    pipe.return_rank()
    pipe.return_ranks(.9,printtofile=True)

###############################################################

    pipe = pipeline.Pipe(X,Y,names,wids)
    pipe.setpipe(['PCA','FFS'])

    # cvcounter test
    # test initialization of grid parameters

    griddic = dict(FFS__k=range(10,501,5),PCA__n_components=range(10,501,5))
    pipe.crossgrid(griddic,cv=cv.leave_x_out(pipe.Y, 10, nsamples=200, testlst=[i for i,n in enumerate(ns) if ('4' in n or '5' in n)]))

    pipe.return_rank()
    pipe.return_ranks(.9,printtofile=True)
