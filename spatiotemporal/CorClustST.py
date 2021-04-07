'''Correlation-based clustering of spatio-temporal data'''

import numpy as np
import pandas as pd
import geopandas as gpd
from warnings import warn
from sklearn.base import BaseEstimator, ClusterMixin

class CorClustST(BaseEstimator, ClusterMixin):
    '''
    CorClustST clustering of spatio-temporal data.
    Reference: Hüsch, M., Schyska, B. U., & von Bremen, L. (2018). CorClustST — Correlation-based clustering of big spatio-temporal datasets. Future Generation Computer Systems.

    Parameters
    ----------
    corr_method: str, default="pearson"
        Correlation method, it has to be one of "pearson", "kendall", "spearman".
    rho: float, default=0.7
        Minimum (absolute) correlation for points to be considered neighbors. The value needs to be between 0.0 and 1.0.
    crs: str, default=None
        Coordinate Reference System. Please indicate one if it is not already present in the data.
    epsilon: float or int, default=None
        Maximum distance for points to be considered neighbors. This value needs to be provided according to the distance measure used for the respective Coordinate Reference System.

    Attributes
    ----------
    corr_method: str
        Correlation method.
    rho: float
        Minimum (absolute) correlation for points to be considered neighbors.
    crs: str
        Coordinate Reference System.
    epsilon: float or int
        Maximum distance for points to be considered neighbors.
    labels_: pandas.Series
        Labels of each point.
    cluster_centers_: pandas.Series
        Label of cluster centers.
    correlation_: pandas.DataFrame
        Correlation matrix among points.
    spatial_distance_: pandas.DataFrame
        Distance matrix among points.
    '''
    
    def __init__(self, corr_method = "pearson", rho = 0.7, crs = None, epsilon=None):
        '''
        Initialize self.
        '''
        
        # corr_method
        if corr_method in ['pearson', 'kendall', 'spearman']:
            self.corr_method = corr_method
        else:
            raise ValueError('The parameter corr_method needs to be one of "pearson", "kendall", "spearman".')
        # rho
        if isinstance(rho, float):
            if 0.0 <= rho <= 1.0:
                self.rho = rho
            else:
                raise ValueError('The parameter rho needs to be a float between 0.0 and 1.0 because it corresponds to the correlation.')
        else:
            raise TypeError('The parameter rho needs to be a float between 0.0 and 1.0 because it corresponds to the correlation.')
        # crs
        self.crs = crs
        # epsilon
        if isinstance(epsilon, float) | isinstance(epsilon, int):
            self.epsilon = epsilon
        else:
            raise TypeError('The parameter epsilon needs to be of numeric type, since it refers to spatial distance.')
    
    def fit(self, X, y=None):
        '''
        Compute CorClustST clustering.     
        
        Parameters
        ----------
        X: geopandas.geodataframe.GeoDataFrame
            Points to cluster. Rows correspond to the points, while columns to the time + geometry column.
        y: Ignored
            Not used, present here for API consistency by convention.
            
        Returns
        -------
        self
            Fitted estimator.
        '''
        # X
        if not isinstance(X, gpd.geodataframe.GeoDataFrame):
            raise TypeError('X needs to be a "GeoDataFrame" to calculate distance between points.')
        # Coordinate Reference System
        if X.crs is None:
            try:
                X.set_crs(self.crs)
            except:
                print('No CRS present or indicated. Please make sure that the value of epsilon works with the distance units.')   
        else:
            if self.crs is not None:
                X.to_crs(self.crs)
                
        #############################################################
        # Step 1: Find all spatio-temporal neighbors for each point #
        #############################################################
        # Calculate spatial distances
        distances = X.geometry.apply(lambda x: X.distance(x))
        # Calculate feature distances (using correlation)
        X = pd.DataFrame(X.drop(columns=X.geometry.name))
        X = X.T
        correlation = X.corr(method=self.corr_method)
        # Find spatio-temporal neighbors as intersection between spatial and temporal neighbors
        sp_neighbors = distances.apply(lambda x: np.where((x <= self.epsilon) & (x > 0)), axis=1, result_type='expand')
        sp_neighbors.columns = ['sp']
        t_neighbors = correlation.apply(lambda x: np.where((abs(x) >= self.rho) & (x < 1)), axis=1, result_type='expand')
        t_neighbors.columns = ['t']
        st_neighbors = pd.concat([sp_neighbors, t_neighbors], axis=1)
        st_neighbors['st'] = st_neighbors.apply(lambda x: [i for i in x['sp'] if i in x['t']], axis=1)
        st_neighbors['neighbors'] = st_neighbors.apply(lambda x: [correlation.index[i] for i in x['st']], axis=1)
        # Calculate the number of spatio-temporal neighbors for each point and order dataframe in descending order.
        st_neighbors['n_neighbors'] = st_neighbors.apply(lambda x: len(x['st']), axis=1)
        st_neighbors.sort_values(by='n_neighbors', ascending=False, inplace=True)
        
        #####################################################################################################
        # Steps 2-4: Assign point as center and neighbors to the cluster if they satisfy certain conditions #
        #####################################################################################################
        # Prepare dataframe
        clusters = st_neighbors[['neighbors', 'n_neighbors']].copy()
        clusters['cluster'] = np.nan
        clusters['cluster_center'] = np.nan
        # Assign clusters
        c = 0
        for p in clusters.index:
            # Is point already assigned to a cluster?
            not_in_cluster = np.isnan(clusters.loc[p, 'cluster'])
            # Are more than 50% of its neighbors not already belong to a cluster?
            p_neighbors = clusters.loc[p, 'neighbors']
            half_not_in_cluster = np.mean(np.isnan(clusters.loc[p_neighbors, 'cluster'])) >= 0.5
            # If point satisfies these qualities, it is assigned as center of a new cluster
            if not_in_cluster & half_not_in_cluster:
                clusters.loc[p, 'cluster'] = c
                clusters.loc[p, 'cluster_center'] = c

                # Which of its neighbors are not yet assigned to a cluster?
                neighbors_not_assigned = np.isnan(clusters.loc[p_neighbors, 'cluster']).loc[lambda x: x==True].index.to_list()
                if len(neighbors_not_assigned) > 0:
                    clusters.loc[neighbors_not_assigned, 'cluster'] = c 

                # Which of the neighbors that already belong to a cluster have a higher correlation with the new center with respect to the old one?
                neighbors_already_assigned = clusters.loc[[x for x in p_neighbors if x not in neighbors_not_assigned]]
                if len(neighbors_already_assigned) > 0:
                    neighbors_already_assigned['cluster_center'] = neighbors_already_assigned.apply(lambda x: clusters.index[clusters['cluster_center'] == x['cluster']].item(), axis=1)
                    neighbors_already_assigned['old_corr'] = neighbors_already_assigned.apply(lambda x: correlation.loc[x.name, x['cluster_center']], axis=1)
                    neighbors_already_assigned['new_corr'] = correlation.loc[p, neighbors_already_assigned.index]
                    neighbors_higher_correlation = (neighbors_already_assigned['new_corr'] > neighbors_already_assigned['old_corr']).loc[lambda x: x==True].index.to_list()  
                    if len(neighbors_higher_correlation) > 0:
                        clusters.loc[neighbors_higher_correlation, 'cluster'] = c

                # If a new cluster has been assigned, increase cluster number
                c += 1
        
        ###############################################################################
        # Step 5: Assigned unclustered points with neighbors according to correlation #
        ###############################################################################
        # If points are not yet assigned and have neighbors, assign them to the cluster of the neighbors with highest correlation.
        for p in clusters[np.isnan(clusters.cluster)].index:
            p_neighbors = clusters.loc[p, 'neighbors']
            if len(p_neighbors) > 0:
                neighbor_max_corr = correlation.loc[p, p_neighbors].idxmax()
                clusters.loc[p, 'cluster'] = clusters.loc[neighbor_max_corr, 'cluster']
        
        #################################################################
        # Step 6: If still not assigned, define as "noise" (label = -1) #
        #################################################################
        clusters.loc[np.isnan(clusters.cluster), 'cluster'] = -1
        
        ##########
        # Result #
        ##########
        self.labels_ = clusters['cluster'].astype('int')
        self.cluster_centers_ = clusters.loc[~np.isnan(clusters.cluster), 'cluster_center'].astype('int')
        self.correlation_ = correlation
        self.spatial_distance_ = distances
        return self
    
    def fit_predict(self, X, y=None):
        '''
        Predict cluster labels for the points and also define cluster centers.
        This is a convenient method, it is equivalent to calling fit(X) followed by predict().
        
        Parameters
        ----------
        X: geopandas.geodataframe.GeoDataFrame
            Points to cluster. Rows correspond to the points, while columns to the time + geometry column.
        y: Ignored
            Not used, present here for API consistency by convention.
            
        Returns
        -------
        labels: pandas.Series
            Labels of each point. 
        '''
        
        self.fit(X)
        return self.labels_
