#! /usr/bin/env python

"""
Copyright of the program: Andrea Agazzi, UNIGE
                            Vincent Deo, Stanford University


"""
import matplotlib.pyplot as plt

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
import sys
import scipy.stats

def function1(n):
        wids = ['week_4','week_5','week_6','week_10']
        ns, Xdata, Ydata = src.importd.importfile('data/file.dat')
        ns, X, Y, y2 = src.importd.filterd(ns,Xdata,Ydata,wids)
        _, names = src.importd.import_cnames('data/file3.dat')

        boxplotlst = [[],[],[],[],[],[],[],[]]

        testlst = [[],[]]

        for i,j in enumerate(y2):
            boxplotlst[j].append(X[i][n])

            if j==0:
                testlst[0].append(X[i][n])
            elif j == 4:
                testlst[1].append(X[i][n])

        print(testlst)
        print(scipy.stats.ranksums(testlst[0],testlst[1]))
        print(scipy.stats.ttest_ind(testlst[0],testlst[1]))


        plt.boxplot(boxplotlst)
        plt.show()


if __name__ == '__main__':

    filename = './results/ranks/FFS_L1LogReg_2016_03_22_20_45_0.dat'
    resultlst, ranklst, _, paramlst = src.importd.import_results(filename)
    function1(120)
    for i,lst in enumerate(resultlst):

        if i == 0:
            plt.figure(i)
            plt.subplot(111)
            plt.title(paramlst[i])
            plt.hist(lst)
            plt.show()
            for j in ranklst[i][0:10]:
                function1(j)
