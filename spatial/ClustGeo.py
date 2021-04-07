'''Hierarchical clustering with geographical contraints'''

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial.distance import cdist, squareform
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.cluster import AgglomerativeClustering
from _distance_matrix import distance_matrix
from _ward_aggr import ward_aggr
from _within_diss import within_diss

class ClustGeo(BaseEstimator, ClusterMixin):
    '''
    ClustGeo: hierarchical clustering with spatial constraints.
    Python implementation of the R package ClustGeo.
    References: 
    - Chavent, M., Kuentz-Simonet, V., Labenne, A., & Saracco, J. (2018). ClustGeo: an R package for hierarchical clustering with spatial constraints. Computational Statistics, 33(4), 1799-1822.
    - Marie Chavent, Vanessa Kuentz, Amaury Labenne and Jerome Saracco (2017). ClustGeo: Hierarchical Clustering with Spatial Constraints. R package version 2.0. https://CRAN.R-project.org/package=ClustGeo
    
    Parameters
    ----------
    alpha: float, default=0
        Mixing parameter between D0 and D1.
    metric: str, default='euclidean'
        Which metric to use to calculate feature distance (if provided object is DataFrame or GeoDataFrame).
        It needs to be one of the accepted values from scipy.spatial.distance.cdist.
        The only alternative is the default 'precomputed', in the case where the provided data is already in the form of distance matrix.
    
    Attributes
    ----------
    alpha: float, default=0
        Mixing parameter between D0 and D1.
    metric: str, default='precomputed'
        Metric for feature distance.
    tree_: ndarray
        The hierarchical clustering encoded as a linkage matrix.
    explained_inertia_: DataFrame
        Proportion and  normalized proportion of explained inertia of the partitions in n_clusters clusters.
    labels_: ndarray of shape (n_samples)
        Cluster labels for each point
    n_clusters_: int
        The number of clusters.
    '''
        
    def __init__(self, alpha=0, metric='precomputed'):
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
        # metric: Either "precomputed" or one accepted by scipy.spatial.distance.cdist
        if metric != 'precomputed':
            try:
                cdist(XA=[[1, 1], [2, 2]], XB=[[1, 1], [2, 2]], metric=metric)
                self.metric = metric
            except:
                raise ValueError('The parameter "metric" needs to be either "precomputed" or one accepted by scipy.spatial.distance.cdist')
        else:
            self.metric = metric
    
    def fit(self, X, y=None, scale=False, wt=None):
        '''
        Compute the entire ClustGeo clustering tree.     
        
        Parameters
        ----------
        X: GeoDataFrame, DataFrame or dictionary
            Object with either original data or already calculated distance matrices.
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
            # Calculate feature distances (using correlation)
            X = pd.DataFrame(X.drop(columns=X.geometry.name))
            if self.metric=='precomputed':
                raise ValueError('metric cannot be "precomputed" with a GeoDataFrame.')
            else:
                D0 = distance_matrix(X, metric=self.metric)
        elif isinstance(X, pd.core.frame.DataFrame):
            # Calculate distance matrix if not already calculated
            if self.metric=='precomputed':
                D0 = X.copy()
            else:
                D0 = distance_matrix(X, metric=self.metric)
            D1 = None
        elif isinstance(X, dict):
            # Precomputed distance matrices
            if sorted(list(X.keys())) == ['D0', 'D1']:
                D0 = X['D0']
                D1 = X['D1']
            else:
                raise ValueError('The dictionary needs to have two elements with keys "D0" and "D1"')
        else:
            raise TypeError('X needs to be either a DataFrame, a GeoDataFrame, or a dictionary with two elements with keys "D0" and "D1"')
        
        ########################################
        # Calculation of dissimilarity measure #
        ########################################
        delta0 = ward_aggr(D0, wt=wt)
        if D1 is not None:
            if scale:
                #Normalized dissimilarity matrices 
                D0 = D0/D0.to_numpy().max()
                D1 = D1/D1.to_numpy().max()       
            delta1 = ward_aggr(D1, wt=wt)
        else:
            delta1 = 0
        delta = (1-self.alpha)*delta0 + self.alpha*delta1
        delta = squareform(X=delta, force='tovector')
        
        ##########################################
        # Clustering with dissimiilarity measure #
        ##########################################
        self.tree_ = linkage(y=delta, method='ward', metric=None)
        return self
    
    def predict(self, X=None, y=None, n_clusters=None, max_dist=None):
        '''
        Return cluster labels for fitted tree according to either number of clusters or maximum distance.
        
        Parameters
        ----------
        X: Ignored
            Not used, present here for API consistency by convention.
        y: Ignored
            Not used, present here for API consistency by convention.
        n_clusters: int, default=None
            Number of clusters. Either one of this or max_dist needs to be different from None.
        max_dist: float, default=None
            Distance where to cut tree. Either one of this or n_clusters needs to be different from None.
        Returns
        -------
        labels: ndarray
            Labels of each point.
        '''
        
        if X is not None:
            raise ValueError('X is ignored, thus needs to be left X=None.') 
            
        if (n_clusters is None) & (max_dist is None):
            self.labels_ = fcluster(Z=self.tree_, t=0.7*max(self.tree_[:,2]), criterion='distance')-1 # to have zero indexing
            self.n_clusters_ = len(np.unique(self.labels_))
        elif n_clusters is not None:
            self.labels_ = fcluster(Z=self.tree_, t=n_clusters, criterion='maxclust')-1 # to have zero indexing
            self.n_clusters_ = n_clusters
        else:
            self.labels_ = fcluster(Z=self.tree_, t=max_dist, criterion='distance')-1 # to have zero indexing
            self.n_clusters_ = len(np.unique(self.labels_))
        return self.labels_
    
    def fit_predict(self, X, y=None, **kwargs):
        fitted = self.fit(X=X, **kwargs)
        return fitted.predict(X=None, **kwargs)
#     def fit_predict(self, X, y=None, scale=False, wt=None, n_clusters=None, max_dist=None):
#         fitted = self.fit(X=X, scale=scale, wt=wt)
#         return fitted.predict(X=None, n_clusters=n_clusters, max_dist=max_dist)
        '''
        Predict cluster labels for the data points.
        This is a convenient method, it is equivalent to calling fit(X) followed by predict().
        
        Parameters
        ----------
        X: GeoDataFrame, DataFrame or dictionary
            Object with either original data or already calculated distance matrices.
        y: Ignored
            Not used, present here for API consistency by convention.
        scale: boolean, default=True
            Whether to scale the distance matrices D0 and D1.
        wt: ndarray, default=None
            Observation weights. If None, all have the same weight 1/#observations.
        n_clusters: int, default=None
            Number of clusters. Either one of this or max_dist needs to be different from None.
        max_dist: float, default=None
            Distance where to cut tree. Either one of this or n_clusters needs to be different from None.
            
        Returns
        -------
        labels: ndarray
            Labels of each point. 
        '''
    
    def alpha_choice(self, X, n_clusters, scale=True, wt=None,
                    range_alpha = np.round(np.linspace(0,1,11,endpoint=True), 2),
                    plot=True):
        '''
        Compute the proportion and  normalized proportion of explained inertia of the partitions in a certain number of clusters for a range of mixing parameters alpha.  
        The plot of these criteria can help the user in the choice of the mixing parameter alpha, which should be a value that balances the decrease in explained inertia for D0 and the increase for D1.

        Parameters
        ----------
        n_clusters: int
            Number of clusters to partition the data.
        scale: boolean, default=True
            Whether to scale the distance matrices D0 and D1.
        wt: ndarray, default=None
            Observation weights. If None, all have the same weight 1/#observations.
        range_alpha: ndarray, default=[0, 0.1, ..., 1]
            A vector of real values between 0 and 1 to test for the mixing parameter alpha.
        plot: boolean, default=True
            Whether to plot the explained inertia or return the DataFrame with the explained inertia.
        
        Returns
        ----------
        Explained inertia plot or DataFrame, depending on parameter plot.
        '''

        if isinstance(X, gpd.geodataframe.GeoDataFrame):
            # Calculate spatial distances
            D1 = X.geometry.apply(lambda x: X.distance(x))
            # Calculate feature distances (using correlation)
            X = pd.DataFrame(X.drop(columns=X.geometry.name))
            if self.metric=='precomputed':
                raise ValueError('metric cannot be "precomputed" with a GeoDataFrame.')
            else:
                D0 = distance_matrix(X, metric=self.metric)
        elif isinstance(X, dict):
            # Precomputed distance matrices
            D0 = X['D0']
            D1 = X['D1']
            
        if scale:
            #Normalized dissimilarity matrices 
            D0 = D0/D0.to_numpy().max()
            D1 = D1/D1.to_numpy().max()
            
        self.n_clusters_ = n_clusters
        
        # Within-cluster inertia obtained either with D0 or D1
        W = pd.DataFrame(index=pd.Index(data=range_alpha, name='alpha'), columns=['D0', 'D1'])
        for i in range_alpha:
            labels=ClustGeo(alpha=i, metric='precomputed').\
                        fit_predict(X={'D0':D0, 'D1': D1}, scale=scale, wt=wt, n_clusters=n_clusters)
            W.loc[i, 'D0'] = within_diss(D=D0, labels=labels, wt=wt)
            W.loc[i, 'D1'] = within_diss(D=D1, labels=labels, wt=wt)
        # Total inertia obtained with either with D0 or D1
        T0 = inert_diss(D=D0, wt=wt)
        T1 = inert_diss(D=D1, wt=wt)
        
        # (Normalized) Proportion of explained inertia obtained either with D0 or D1
        QQ = pd.DataFrame(index=pd.Index(data=range_alpha, name='alpha'),
                 columns=pd.MultiIndex.from_product([['Q', 'Qnorm'], ['D0','D1']]))
        QQ[('Q','D0')] = 1-W['D0']/T0
        QQ[('Q','D1')] = 1-W['D1']/T0
        QQ[('Qnorm','D0')] = QQ[('Q','D0')]/QQ[('Q','D0')].iloc[0]
        QQ[('Qnorm','D1')] = QQ[('Q','D1')]/QQ[('Q','D1')].iloc[-1]
        self.explained_inertia_ = QQ
        
        if plot:
            QQ_pivot = QQ.stack().stack()\
                        .rename_axis(['alpha', 'D', 'Q_type'])\
                        .reset_index(name='value')
            QQ_pivot['Q_type'] = QQ_pivot['Q_type'].replace({'Q': 'proportion', 'Qnorm':'normalized proportion'})
            g = sns.catplot(kind='point', data=QQ_pivot, x='alpha', y='value', 
                            hue='D', col='Q_type', palette={'D0':'black', 'D1':'red'}, 
                            legend_out=True)
            g.set_titles('{col_name}')
            g.fig.subplots_adjust(top=0.88)
            g.fig.suptitle('Empirical choice of the mixing parameter for ' + 
                           str(n_clusters) + ' clusters')
            g.set_axis_labels('alpha','proportion of explained pseudo-inertias')
            g.legend.set_title('based on')
            g.set(ylim=(-0.05, 1.05))
            ax = g.axes[0,1]
            D0_Qnorm = int(QQ_pivot['value'][(QQ_pivot['alpha'] == 0.0) & (QQ_pivot['D'] == 'D0') \
                        & (QQ_pivot['Q_type'] =='normalized proportion')].item()*100)
            D1_Qnorm = int(QQ_pivot['value'][(QQ_pivot['alpha'] == 0.0) & (QQ_pivot['D'] == 'D1') \
                        & (QQ_pivot['Q_type'] =='normalized proportion')].item()*100)
            ax.text(0, D0_Qnorm/100-0.07, 
                    'with D0 & alpha=0: '+str(D0_Qnorm)+'%', 
                    horizontalalignment='left', color='black', size='small')
            ax.text(len(range_alpha)-1, D1_Qnorm/100-0.07, 
                    'with D1 & alpha=1: '+str(D1_Qnorm)+'%', 
                    horizontalalignment='right', color='red', size='small')
            plt.show()
        else:
            return self.explained_inertia_