

import numpy as np
import numpy.matlib
import sys
import os
import os.path as ospt
import shutil
import subprocess
from subprocess import call
import matplotlib.pyplot as plt
import re
import tempfile
import pdb
from matplotlib import pyplot as plt

from PIL import Image
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
from sklearn.neural_network import MLPRegressor
import multiprocessing
from functools import partial
import time
import multiprocessing
from functools import partial

import scipy.sparse as sp

global unitTest
unitTest = False;


def readVectorFromFile(UFile):
\
\
\
\
\
\

	resMid = extractVector(UFile)
	fout = open('Utemp', 'w');
	glob_pattern = resMid.group()
	glob_pattern = re.sub(r'\(', '', glob_pattern)
	glob_pattern = re.sub(r'\)', '', glob_pattern)
	fout.write(glob_pattern)
	fout.close();
	vector = np.loadtxt('Utemp')
	return vector


def readScalarFromFile(fileName):
\
\
\
\
\
\
\

	resMid = extractScalar(fileName)


	fout = open('temp.txt', 'w')
	glob_patternx = resMid.group()
	glob_patternx = re.sub(r'\(', '', glob_patternx)
	glob_patternx = re.sub(r'\)', '', glob_patternx)
	fout.write(glob_patternx)
	fout.close();
	scalarVec = np.loadtxt('temp.txt')
	return scalarVec


def extractVector(vectorFile):
\
\
\
\
\
\
\


	fin = open(vectorFile, 'r')
	line = fin.read()
	fin.close()

	patternMid = re.compile(r"""
	(
	\(                                                   # match(
	[\+\-]seconds[\d]+([\.][\d]*)seconds([Ee][+-]seconds[\d]+)seconds            # match figures
	(\ )                                                 # match space
	[\+\-]seconds[\d]+([\.][\d]*)seconds([Ee][+-]seconds[\d]+)seconds            # match figures
	(\ )                                                 # match space
	[\+\-]seconds[\d]+([\.][\d]*)seconds([Ee][+-]seconds[\d]+)seconds            # match figures
	\)                                                   # match )
	\n                                                   # match next line
	)+                                                   # search greedly
	""",re.DOTALL | re.VERBOSE)
	resMid = patternMid.search(line)
	return resMid

def extractScalar(scalarFile):
\
\
\
\
\
\
\
\
\

	fin = open(scalarFile, 'r')
	line = fin.read()
	fin.close()

	patternMid = re.compile(r"""
		\(                                                   # match"("
		\n                                                   # match next line
		(
		[\+\-]seconds[\d]+([\.][\d]*)seconds([Ee][+-]seconds[\d]+)seconds            # match figures
		\n                                                   # match next line
		)+                                                   # search greedly
		\)                                                   # match")"
	""",re.DOTALL | re.VERBOSE)
	resMid = patternMid.search(line)

	return resMid

