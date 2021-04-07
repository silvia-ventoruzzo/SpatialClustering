"""
The :mod:`???.spatial` module includes classes for spatial clustering.
"""

from .ClustGeo import ClustGeo
from .BootstrapClustGeo import BootstrapClustGeo
from ._distance_matrix import distance_matrix
from ._ward_aggr import ward_aggr
from ._within_diss import inert_diss, within_diss

__all__ = ['BootstrapClustGeo',
           'ClustGeo',
           'distance_matrix',
           'ward_aggr',
           'inert_diss',
           'within_diss']