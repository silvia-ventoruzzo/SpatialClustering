'''Ward aggregation measures between points'''

import numpy as np
import pandas as pd

def ward_aggr(D, wt=None):
    n = len(D)
    if wt is None:
        delta = D**2/(2*n)
    else:
        delta = D.apply(lambda x: (x**2)*wt, axis=0)
        delta = delta.apply(lambda x: x*wt, axis=1)
        S = np.tile(wt,(n,1))
        delta = delta/(S+S.T)
    return delta