## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - A Synoptic Framework for Standardized Machine Learning Tasks
## -- Package : mlpro.oa.tasks
## -- Module  : clusteranalyzers.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2023-01-24  0.0.0     DA       Creation
## -- 2023-04-18  0.1.0     DA       First implementation of classes ClusterMembership, ClusterAnalyzer
## -------------------------------------------------------------------------------------------------

"""
Ver. 0.1.0 (2023-04-18)

This module provides templates for cluster analysis to be used in the context of online adaptivity.
"""

from mlpro.oa.streams import *
from mlpro.bf.math.geometry import Point
from typing import List, Tuple






## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class Cluster (Id, Plottable):
    """
    Base class for a cluster. 

    Parameters
    ----------
    p_visualize : bool
        Boolean switch for visualisation. Default = False.
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
            Scalar value that determines the membership of the given instance to this cluster. A
            value 0 means that the given instance is not a member of the cluster.
        """

        raise NotImplementedError





## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class ClusterAnalyzer (OATask):
    """
    Base class for online cluster analysis. It raises an event when a cluster was added or removed.

    Parameters
    ----------
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
    p_cls_cluster 
        Cluster class (Class Cluster or a child class). Default = Cluster.
    p_kwargs : dict
        Further optional named parameters.
    """

    C_TYPE                  = 'Cluster Analyzer'

    C_EVENT_CLUSTER_ADDED   = 'CLUSTER_ADDED'
    C_EVENT_CLUSTER_REMOVED = 'CLUSTER_REMOVED'

    C_PLOT_ACTIVE           = True
    C_PLOT_STANDALONE       = False

## -------------------------------------------------------------------------------------------------
    def __init__( self, 
                  p_name: str = None, 
                  p_range_max = StreamTask.C_RANGE_THREAD, 
                  p_ada: bool = True, 
                  p_duplicate_data: bool = False, 
                  p_visualize: bool = False, 
                  p_logging = Log.C_LOG_ALL, 
                  p_cls_cluster = Cluster,
                  **p_kwargs ):
        
        self._cls_cluster = p_cls_cluster
        self._clusters    = {}

        super().__init__( p_name = p_name, 
                          p_range_max = p_range_max, 
                          p_ada = p_ada, 
                          p_duplicate_data = p_duplicate_data, 
                          p_visualize = p_visualize, 
                          p_logging = p_logging, 
                          **p_kwargs )


## -------------------------------------------------------------------------------------------------
    def _run(self, p_inst_new: List[Instance], p_inst_del: List[Instance]):
        self.adapt( p_inst_new=p_inst_new, p_inst_del=p_inst_del )


## -------------------------------------------------------------------------------------------------
    def get_clusters(self) -> List[Cluster]:
        """
        This method returns the current list of clusters. 

        Returns
        -------
        list_of_clusters : List[Cluster]
            Current list of clusters.
        """

        return self._clusters.values()
    

## -------------------------------------------------------------------------------------------------
    def get_cluster_membership(self, p_inst : Instance ) -> List[Tuple[str, float, Cluster]]:
        """
        Public custom method to determine the membership of the given instance to each cluster as
        a value in percent.

        Parameters
        ----------
        p_inst : Instance
            Instance to be evaluated.

        Returns
        -------
        membership : List[Tuple[str, float, Cluster]]
            List of membership tuples for each cluster. A tuple consists of a cluster id, a
            relative membership value in percent and a reference to the cluster.
        """

        clusters        = self.get_clusters()
        sum_memberships = 0
        memberships_abs = []
        memberships_rel = []

        for cluster in clusters:
            membership_abs = cluster.get_membership( p_inst = p_inst )
            memberships_abs.append(membership_abs)
            sum_memberships += membership_abs

        if sum_memberships > 0:
            for cluster in clusters:
                membership_rel = memberships_abs.pop(0) / sum_memberships
                memberships_rel.append( ( cluster.get_id(), membership_rel, cluster) )
        else:
            for cluster in clusters:
                memberships_rel.append( ( cluster.get_id(), 0, cluster) )

        return memberships_rel


## -------------------------------------------------------------------------------------------------
    def init_plot(self, p_figure: Figure = None, p_plot_settings: PlotSettings = None):
        Plottable.init_plot(self, p_figure, p_plot_settings)


## -------------------------------------------------------------------------------------------------
    def _init_plot_2d(self, p_figure: Figure, p_settings: PlotSettings): pass


## -------------------------------------------------------------------------------------------------
    def _init_plot_3d(self, p_figure: Figure, p_settings: PlotSettings): pass


## -------------------------------------------------------------------------------------------------
    def _init_plot_nd(self, p_figure: Figure, p_settings: PlotSettings): pass
        
    
## -------------------------------------------------------------------------------------------------
    def update_plot( self, 
                     p_inst_new: List[Instance] = None, 
                     p_inst_del: List[Instance] = None, 
                     **p_kwargs ):

        if not self.get_visualization(): return

        for cluster in self._clusters.values():
            cluster.update_plot(p_inst_new = p_inst_new, p_inst_del = p_inst_del, **p_kwargs)





## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class ClusterCentroid (Cluster):
    """
    Extended cluster class with a centroid.
    """

## -------------------------------------------------------------------------------------------------
    def __init__( self, 
                  p_id=None, 
                  p_visualize: bool = False, 
                  p_centroid_pos : Element = None,
                  **p_kwargs ):
        
        self._centroid : Point = Point( p_pos=p_centroid_pos )

        super().__init__( p_id = p_id, p_visualize = p_visualize, **p_kwargs )


## -------------------------------------------------------------------------------------------------
    def get_centroid(self) -> Point:
        return self._centroid
    

## -------------------------------------------------------------------------------------------------
    def set_centroid_pos(self, p_centroid_pos : Element):
        pass


## -------------------------------------------------------------------------------------------------
    def get_membership(self, p_inst: Instance) -> float:
        return super().get_membership(p_inst)
