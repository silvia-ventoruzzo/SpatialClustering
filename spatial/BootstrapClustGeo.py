'''Bootstrap hierarchical clustering with geographical contraints'''

import numpy as np
import pandas as pd
import geopandas as gpd
from warnings import warn
from _distance_matrix import distance_matrix
from ClustGeo import ClustGeo

class BootstrapClustGeo(ClustGeo):
    '''
    BootstrapClustGeo: bootstrap hierarchical clustering with spatial constraints.
    Python implementation of the the clustering technique Bootstrap ClustGeo.
    References: 
    - Distefano, V., Mameli, V., & Poli, I. (2020). Identifying spatial patterns with the Bootstrap ClustGeo technique. Spatial Statistics, 38, 100441.
    
    Parameters
    ----------
    alpha: float, default=0
        Mixing parameter between D0 and D1.
    metric: str, default='euclidean'
        Which metric to use to calculate feature distance.
        It needs to be one of the accepted values from scipy.spatial.distance.cdist.
    n_bootstraps: int, default=500
        Number of bootstrap repetitions.
    range_clusters: array, list, range, default=None
        Possible numbers of clusters. If None, it will range from 2 to the square root of the number of points.
    b_type: int, default=1
        How many times the features will be sampled. b_type=1 is suggested when the number of points is higher than the number of features, b_type=2 in the opposite case.
    
    Attributes
    ----------
    alpha: float, default=0
        Mixing parameter between D0 and D1.
    metric: str, default='euclidean'
        Which metric to use to calculate feature distance.
        It needs to be one of the accepted values from scipy.spatial.distance.cdist.
    n_bootstraps: int, default=500
        Number of bootstrap repetitions.
    range_clusters: array, list, range, default=None
        Possible numbers of clusters. If None, it will range from 2 to the square root of the number of points.
    b_type: int, default=1
        How many times the features will be sampled. b_type=1 is suggested when the number of points is higher than the number of features, b_type=2 in the opposite case.
    tree_: ndarray
        The hierarchical clustering encoded as a linkage matrix.
    explained_inertia_: DataFrame
        Proportion and  normalized proportion of explained inertia of the partitions in n_clusters clusters.
    labels_: ndarray of shape (n_samples)
        Cluster labels for each point
    n_clusters_: int
        The number of clusters.
    '''
        
    def __init__(self, alpha=0, metric='euclidean', 
                 n_bootstraps=500, range_clusters = None, b_type=1):
        '''
        Initialize self.
        '''

        # alpha
        if isinstance(alpha, float) | isinstance(alpha, int):
            if 0 <= alpha <= 1:
                self.alpha = alpha
            else:
                raise ValueError('The parameter alpha needs to be a number between 0 and 1.')
        else:
            raise TypeError('The parameter alpha needs to be a number between 0 and 1.')
        # metric: One accepted by scipy.spatial.distance.cdist
        try:
            cdist(XA=[[1, 1], [2, 2]], XB=[[1, 1], [2, 2]], metric=metric)
            self.metric = metric
        except:
            raise ValueError('The parameter "metric" needs to be either "precomputed" or one accepted by scipy.spatial.distance.cdist')
        # n_bootstraps: number of bootstraps needs to be a positive and finite integer
        if isinstance(n_bootstraps, int) & n_bootstraps>0 & np.isfinite(n_bootstraps):
            self.n_bootstraps = n_bootstraps
        else:
            raise ValueError('The parameter "n_bootstraps" needs to be a positive and finite integer.')
        # range clusters
        if range_clusters is None:
            self.range_clusters = range(2, int(np.sqrt(N)))
            
        range_clusters = np.array(range_clusters)
        elif issubclass(range_clusters.dtype.type, np.integer) & (range_clusters>0).all() & np.isfinite(range_clusters).all():
            self.range_clusters = range_clusters
        else:
            raise ValueError('The parameter "range_clusters" needs to contain only positive and finite integers.')
        # b_type
        if b_type in [1, 2]:
            self.b_type = b_type
        else:
            raise ValueError('The parameter "b_type" must be either 1 or 2')
    
    def fit(self, X, y=None, scale=False, wt=None):
        '''
        Compute the bootstrapping process and obtain a ClustGeo tree.     
        
        Parameters
        ----------
        X: GeoDataFrame or DataFrame
            Object with either original data.
        y: Ignored
            Not used, present here for API consistency by convention.
        scale: boolean, default=True
            Whether to scale the distance matrices D0 and D1.
        wt: ndarray, default=None
            Observation weights. If None, all have the same weight 1/#observations.
            
        Returns
        -------
        self
            Fitted estimator.
        '''
        
        if isinstance(X, gpd.geodataframe.GeoDataFrame):
            # Calculate spatial distances
            D1 = X.geometry.apply(lambda x: X.distance(x))
            # Remove geometry and transform to DataFrame
            X = pd.DataFrame(X.drop(columns=X.geometry.name))
        elif isinstance(X, pd.core.frame.DataFrame):
            # Spatial distances not considered
            warn('Potential spatial distance not considered since data not provided as GeoDataFrame.')
            D1 = None
        else:
            raise TypeError('X needs to be either a DataFrame or a GeoDataFrame.')
        
        # Number of points (N) and features (M)
        N = len(X)
        M = len(X.columns)
        
        #Normalized dissimilarity matrices 
        if scale:
            D1 = D1/D1.to_numpy().max()
        
        # Bootstrapping
        XB = pd.DataFrame(index=X.index)
        for i in range(self.n_bootstraps):
            # Sample of features
            M_sample = list(set(random.choices(population=range(M), k=M)))
            if self.b_type == 2:
                M_sample = list(set(random.choices(population=M_sample, k=len(M_sample))))
            # Sample of number of clusters
            k_sample = random.choice(seq=self.range_clusters)
            # Distance matrix only with sampled features
            D0_sample = distance_matrix(X=X.iloc[:,M_sample], metric=self.metric)
            if scale:
                #Normalized dissimilarity matrices 
                D0_sample = D0_sample/D0_sample.to_numpy().max()
            # Cluster labels from ClustGeo
            XB[str(i)] = ClustGeo(alpha=self.alpha, metric='precomputed')\
                    .fit_predict(X={'D0':D0_sample, 'D1':D1}, scale=scale, wt=wt, n_clusters=k_sample)
       
        # Hamming distance
        DB = distance_matrix(X=DB, metric='hamming')
        
        # Clustering with the new distance matrix
        self.tree_ = ClustGeo(alpha=alpha, metric='precomputed')\
                        .fit(X={'D0':DB, 'D1':None}, scale=scale, wt=wt)
        return self