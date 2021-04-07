'''Dissimilarity based pseudo within-cluster inertia of a partition'''

import numpy as np
import pandas as np

def inert_diss(D, indices=None, wt=None):
    '''Pseudo inertia of a cluster'''
    n = len(D)
    if indices is None:
        indices = np.array(range(n))
    if wt is None:
        wt = np.repeat(1/n, n)
    if indices.size > 1:
        subD = D.iloc[indices, indices]
        subW = wt[indices]
        mu = sum(subW)
        inert = subD.apply(lambda x: (x**2)*subW, axis=0)
        inert = inert.apply(lambda x: x*subW, axis=1)
        inert = (inert/(2*mu)).to_numpy().sum()
    else:
        inert = 0
    return(inert)

def within_diss(D, labels, wt=None):
    '''Dissimilarity based pseudo within-cluster inertia of a partition'''
    n = len(D)
    k = len(np.unique(labels))
    W = 0
    for i in range(k):
        A = np.where(labels==i)[0]
        W += inert_diss(D=D, indices=A, wt=wt)
    return(W)