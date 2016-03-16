
#! /usr/bin/env python
import sys
import numpy as np
import cmath as math
from functools import reduce


# import data
def importfile(filename):
	"""
	import the training set from filename
	returns
	- nlst (list of sample names)
	- Xdata (matrix of predictors)
	- Ydata (array of outcomes)
	"""
	with open(filename,'r') as datafile:
		for i,line in enumerate(datafile):
			if i == 0:
				l = line.strip().split('\t')
				nlst = l[2:]
				Xdata = [[] for j in range(len(l)-2)]
			else:
				l = map(float,line.strip().split('\t'))
				for j,v in enumerate(l):
					if j not in (0,1):
						Xdata[j-2].append(v)
	Ydata = [1 if 'ctrl' in n else 0 for i,n in enumerate(nlst)]
	print('Import:\tcomplete')
	return nlst, Xdata, Ydata


def import_cnames(filename):
	"""
	import the names AND THE KEGG IDS of the chemical elements corresponding to different features of the predictors
	returns
	- cnlst: a list containing the names of the compounds that have the same mass as the compound ID corresponding to the entry of the array +1
	- cidlst: a list containing the KEGG-IDs of the compounds that have the same mass as the compound ID corresponding to the entry of the array +1
	"""
	# attention: the number 756 here is correspondent to this experiment!
	cnlst = []
	cidlst = []
	with open(filename,'r') as datafile:
		for i,line in enumerate(datafile):
			l = line.strip().split('\t')
			cnlst.append(l[2].strip().split('; ')[0])
			cidlst.append(l[1].strip().split('; ')[0])
	return cidlst, cnlst

def casesn(n):
	"""
	returns a number associated to each of the weeks in the range (1; 4)
	"""
	# TODO change to integer switch
	if '4' in n:
		return 0
	elif '5' in n:
		return 1
	elif '6' in n:
		return 2
	elif '10' in n:
		return 3
	else:
		print('error:'+str(n))
		return None

def filterd(nlst, Xdata, Ydata, wids=['week_4','week_5','week_6','week_10']):
	"""
	Filter of the input data according to weeks.
	input:
	- the input data and the names of the weeks you want to select (in an array)
	output:
	- all the data (Xdata, Ydata, nlst) exactly in the same format as before but filtered according to the desired weeks
	- datas of the different families distinguishing between the different weeks: Ydata2 has value from 0 to 7, where the first 4 numbers correspond to "ko" in the corresponding weeks and the second set of 4 elements corresponds to the "ctrl" of the corresponding weeks.
	"""
	y2 = [4*y+casesn(nlst[i]) for i,y in enumerate(Ydata)]
	selectlst = [reduce(lambda x,y:x or y,[wid in n for wid in wids]) for n in nlst]
	return [n for i,n in enumerate(nlst) if selectlst[i]],np.array([x for i,x in enumerate(Xdata) if selectlst[i]]),np.array([y for i,y in enumerate(Ydata) if selectlst[i]]),np.array([y for i,y in enumerate(y2) if selectlst[i]])
