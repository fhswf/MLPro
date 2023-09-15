## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - A Synoptic Framework for Standardized Machine Learning Tasks
## -- Package : mlpro.oa.tasks
## -- Module  : clusteranalyzers.py
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
## -------------------------------------------------------------------------------------------------

"""
Ver. 0.4.0 (2023-06-03)

This module provides templates for cluster analysis to be used in the context of online adaptivity.
"""

from mlpro.bf.various import *
from mlpro.bf.plot import *
from mlpro.bf.streams import *
from mlpro.oa.streams import OATask
from mlpro.bf.math.geometry import Point
from typing import List, Tuple




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
    **p_kwargs
        Further optional keyword arguments.
    """

    C_PLOT_ACTIVE           = True
    C_PLOT_STANDALONE       = False
    C_PLOT_VALID_VIEWS      = [ PlotSettings.C_VIEW_2D, 
                                PlotSettings.C_VIEW_3D, 
                                PlotSettings.C_VIEW_ND ]
    C_PLOT_DEFAULT_VIEW     = PlotSettings.C_VIEW_ND

## -------------------------------------------------------------------------------------------------
    def __init__( self, 
                  p_id = None,
                  p_visualize : bool = False,
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
## -------------------------------------------------------------------------------------------------
class ClusterAnalyzer (OATask):
    """
    Base class for online cluster analysis. It raises an event when a cluster was added or removed.

    Parameters
    ----------
    p_cls_cluster 
        Cluster class (Class Cluster or a child class).
    p_cluster_limit : int
        Optional limit for clusters to be created. Default = 0 (no limit).
    p_name : str
        Optional name of the task. Default is None.
    p_range_max : int
        Maximum range of asynchonicity. See class Range. Default is Range.C_RANGE_PROCESS.
    p_ada : bool
        Boolean switch for adaptivitiy. Default = True.
    p_duplicate_data : bool
        If True, instances will be duplicated before processing. Default = False.
    p_visualize : bool
        Boolean switch for visualisation. Default = False.
    p_logging
        Log level (see constants of class Log). Default: Log.C_LOG_ALL
    p_kwargs : dict
        Further optional named parameters.

    Attributes
    ----------
    C_MS_SCOPE_ALL : int = 0
        Membership scope, that includes all clusters
    C_MS_SCOPE_NONZERO : int = 1
        Membership scope, that includes just clusters with membership values > 0
    C_MS_SCOPE_MAX : int = 2
        Membership scope, that includes just the cluster with the highest membership value.
    """

    C_TYPE                  = 'Cluster Analyzer'

    C_EVENT_CLUSTER_ADDED   = 'CLUSTER_ADDED'
    C_EVENT_CLUSTER_REMOVED = 'CLUSTER_REMOVED'

    C_PLOT_ACTIVE           = True
    C_PLOT_STANDALONE       = False

    # Possible membership scopes for method get_cluster_memberships
    C_MS_SCOPE_ALL : int    = 0
    C_MS_SCOPE_NONZERO :int = 1
    C_MS_SCOPE_MAX :int     = 2

## -------------------------------------------------------------------------------------------------
    def __init__( self, 
                  p_cls_cluster,
                  p_cluster_limit : int = 0,
                  p_name: str = None, 
                  p_range_max = StreamTask.C_RANGE_THREAD, 
                  p_ada: bool = True, 
                  p_duplicate_data: bool = False, 
                  p_visualize: bool = False, 
                  p_logging = Log.C_LOG_ALL, 
                  **p_kwargs ):
        
        super().__init__( p_name = p_name, 
                          p_range_max = p_range_max, 
                          p_ada = p_ada, 
                          p_duplicate_data = p_duplicate_data, 
                          p_visualize = p_visualize, 
                          p_logging = p_logging, 
                          **p_kwargs )

        self._cls_cluster   = p_cls_cluster
        self._clusters      = []
        self._cluster_limit = p_cluster_limit


## -------------------------------------------------------------------------------------------------
    def _run(self, p_inst_new: List[Instance], p_inst_del: List[Instance]):
        self.adapt( p_inst_new=p_inst_new, p_inst_del=p_inst_del )


## -------------------------------------------------------------------------------------------------
    def new_cluster_allowed(self) -> bool:
        """
        Determines whether adding a new cluster is allowed.

        Returns
        -------
        bool
           True, if adding a new cluster allowed. False otherwise.
        """

        return ( self._cluster_limit == 0 ) or ( len(self._clusters) < self._cluster_limit )
    

## -------------------------------------------------------------------------------------------------
    def get_clusters(self) -> List[Cluster]:
        """
        This method returns the current list of clusters. 

        Returns
        -------
        list_of_clusters : List[Cluster]
            Current list of clusters.
        """

        return self._clusters
    

## -------------------------------------------------------------------------------------------------
    def get_cluster_memberships( self, 
                                 p_inst : Instance,
                                 p_scope : int = C_MS_SCOPE_MAX ) -> List[Tuple[str, float, Cluster]]:
        """
        Method to determine the membership of the given instance to each cluster as a value in 
        percent. 

        Parameters
        ----------
        p_inst : Instance
            Instance to be evaluated.
        p_scope : int
            Scope of the result list. See class attributes C_MS_SCOPE_* for possible values. Default
            value is C_MS_SCOPE_MAX.

        Returns
        -------
        membership : List[Tuple[str, float, Cluster]]
            List of membership tuples. A tuple consists of a cluster id, a relative membership 
            value in [0,1] and a reference to the cluster object.
        """

        # 1 Determination of membership values of the instance for all clusters
        sum_ms          = 0
        list_ms_abs     = []
        cluster_max_ms  = None

        for cluster in self._clusters:

            ms_abs  = cluster.get_membership( p_inst = p_inst )
            sum_ms += ms_abs

            if ( p_scope != self.C_MS_SCOPE_ALL ) and ( ms_abs == 0 ): continue

            if p_scope == self.C_MS_SCOPE_MAX:
                # Cluster with highest membership value is buffered
                if ( cluster_max_ms is None ) or ( ms_abs > cluster_max_ms[1] ):
                    cluster_max_ms = ( cluster, ms_abs )

            else:
                list_ms_abs.append( (cluster, ms_abs) )

        if sum_ms == 0: return []

        if cluster_max_ms is not None:
            ms_abs.append( cluster_max_ms )            


        # 2 Determination of relative membership values according to the required scope
        list_ms_rel = []

        for ms_abs in list_ms_abs:
            ms_rel = ms_abs[1] / sum_ms
            list_ms_rel.append( ( ms_abs[0].get_id(), ms_rel, ms_abs[0] ) )

        return list_ms_rel


## -------------------------------------------------------------------------------------------------
    def update_plot( self, 
                     p_inst_new: List[Instance] = None, 
                     p_inst_del: List[Instance] = None, 
                     **p_kwargs ):

        if not self.get_visualization(): return

        for cluster in self._clusters:
            cluster.update_plot(p_inst_new = p_inst_new, p_inst_del = p_inst_del, **p_kwargs)





## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class ClusterCentroid (Cluster):
    """
    Extended cluster class with a centroid.

    Parameters
    ----------
    p_id
        Optional external id
    p_visualize : bool
        Boolean switch for visualisation. Default = False.
    p_cls_centroid = Point
        Name of a point class. Default = Point
    **p_kwargs
        Further optional keyword arguments.
    """

## -------------------------------------------------------------------------------------------------
    def __init__( self, 
                  p_id = None, 
                  p_visualize : bool = False, 
                  p_cls_centroid = Point,
                  **p_kwargs ):
        
        self._centroid : Point = p_cls_centroid( p_visualize=p_visualize )
        super().__init__( p_id = p_id, p_visualize = p_visualize, **p_kwargs )


## -------------------------------------------------------------------------------------------------
    def get_centroid(self) -> Point:
        return self._centroid
    

## -------------------------------------------------------------------------------------------------
    def get_membership(self, p_inst: Instance) -> float:
        feature_data = p_inst.get_feature_data()
        return feature_data.get_related_set().distance( p_e1 = feature_data, p_e2 = self._centroid )
