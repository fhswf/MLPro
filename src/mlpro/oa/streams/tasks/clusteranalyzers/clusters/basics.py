## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - The integrative middleware framework for standardized machine learning
## -- Package : mlpro.oa.streams.tasks.clusteranalyzers.clusters
## -- Module  : basics.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2023-01-24  0.0.0     DA       Creation
## -- 2023-04-18  0.1.0     DA       First implementation of classes ClusterMembership, ClusterAnalyzer
## -- 2023-05-06  0.2.0     DA       New class ClusterCentroid
## -- 2023-05-14  0.3.0     DA       Class ClusterAnalyzer: simplification
## -- 2023-05-30  0.3.1     DA       Further comments, docstrings
## -- 2023-06-03  0.4.0     DA       Method ClusterAnalyzer.get_cluster_memberships():
## --                                - renaming
## --                                - new parameter p_scope
## --                                - refactoring
## --                                New Method ClusterAnalyzer.new_cluster_allowed()
## -- 2023-11-18  0.5.0     DA       Class ClusterCentroid: added plot functionality
## -- 2023-12-08  0.6.0     DA       Class ClusterAnalyzer: 
## --                                - changed internal cluster storage from list to dictionary
## --                                - added method _remove_cluster()
## -- 2023-12-10  0.6.1     DA       Bugfix in method ClusterAnalyzer.get_cluster_membership()
## -- 2023-12-20  0.7.0     DA       Renormalization
## -- 2024-02-23  0.8.0     DA       Class ClusterCentroid: implementation of methods _remove_plot*
## -- 2024-02-24  0.8.1     DA       Method ClusterAnalyzer._remove_cluster() explicitely removes
## --                                the plot of a cluster before removal of the cluster itself
## -- 2024-02-24  0.8.2     DA       Class ClusterCentroid: redefined method remove_plot()
## -- 2024-04-10  0.8.3     DA       Refactoring
## -------------------------------------------------------------------------------------------------

"""
Ver. 0.8.3 (2024-04-10)

This module provides templates for clusters to be used in cluster analyzer algorithms.
"""

from mlpro.bf.mt import Figure, PlotSettings
from mlpro.bf.various import *
from mlpro.bf.plot import *
from mlpro.bf.streams import *
from mlpro.bf.math.normalizers import Normalizer




## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class Cluster (Id, Plottable):
    """
    Base class for a cluster. 

    Parameters
    ----------
    p_id
        Optional external id.
    p_visualize : bool
        Boolean switch for visualisation. Default = False.
    p_color : string
        Color of the cluster during visualization.
    **p_kwargs
        Further optional keyword arguments.
    """

    C_PLOT_ACTIVE           = True
    C_PLOT_STANDALONE       = False
    C_PLOT_VALID_VIEWS      = [ PlotSettings.C_VIEW_2D, 
                                PlotSettings.C_VIEW_3D, 
                                PlotSettings.C_VIEW_ND ]
    C_PLOT_DEFAULT_VIEW     = PlotSettings.C_VIEW_ND

    C_CLUSTER_COLORS        = [ 'blue', 'orange', 'green', 'red', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan' ]

## -------------------------------------------------------------------------------------------------
    def __init__( self, 
                  p_id = None,
                  p_visualize : bool = False,
                  p_color = 'red',
                  **p_kwargs ):

        self._kwargs = p_kwargs.copy()
        Id.__init__( self, p_id = p_id )
        Plottable.__init__( self, p_visualize = p_visualize )


## -------------------------------------------------------------------------------------------------
    def get_membership(self, p_inst : Instance ) -> float:
        """
        Custom method to compute a scalar membership value for the given instance.

        Parameters
        ----------
        p_inst : Instance
            Instance.

        Returns
        -------
        float
            Scalar value >= 0 that determines the membership of the given instance to this cluster. 
            A value 0 means that the given instance is not a member of the cluster.
        """

        raise NotImplementedError
    

 ## -------------------------------------------------------------------------------------------------
    def renormalize(self, p_normalizer:Normalizer):
        """
        Custom method to renormalize internally buffered data using the given normalizer object. 
        This method is called especially by method ClusterAnalyzer._renormalize().
        
        Parameters
        ----------
        p_normalizer : Normalizer
            Normalizer object to be applied on task-specific 
        """

        pass 
