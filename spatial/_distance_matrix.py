'''Distance matrix'''

import pandas as pd
from scipy.spatial.distance import cdist

def distance_matrix(X, **kwargs):
    index = X.index
    d = pd.DataFrame(cdist(X, X, **kwargs))
    d.index = index
    d.columns = index
    return d