## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - The integrative middleware framework for standardized machine learning
## -- Package : mlpro.oa.streams.tasks.clusteranalyzers
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
## -- 2024-05-04  0.9.0     DA       Introduction of cluster properties
## -------------------------------------------------------------------------------------------------

"""
Ver. 0.9.0 (2024-05-04)

This module provides templates for cluster analysis to be used in the context of online adaptivity.
"""

from matplotlib.figure import Figure
from mlpro.bf.math.properties import *
from mlpro.bf.mt import PlotSettings
from mlpro.bf.streams import Instance
from mlpro.bf.various import *
from mlpro.bf.plot import *
from mlpro.oa.streams import OATask
from mlpro.bf.math.normalizers import Normalizer
from mlpro.oa.streams.tasks.clusteranalyzers.clusters import Cluster
from typing import List, Tuple




## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class ClusterAnalyzer (OATask):
    """
    Base class for online cluster analysis. It raises an event when a cluster was added or removed.

    Steps to implement a new algorithm are:
    - Create a new class and inherit from this base class
    - Specify all cluster properties provided/maintained by your algorithm in C_CLUSTER_PROPERTIES.
    - Implement method self._adapt() to update your cluster list on new instances
    - Implement method self._adapt_reverse() to update your cluster list on obsolete instances
    - New cluster: hand over self._cluster_properties on instantiation
    
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
    C_CLUSTER_PROPERTIES : PropertyDefinitions
        List of cluster properties supported/maintained by the algorithm. These properties 
        are handed over to each new cluster.
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

    # List of cluster properties supported/maintained by the algorithm
    C_CLUSTER_PROPERTIES : PropertyDefinitions = []

## -------------------------------------------------------------------------------------------------
    def __init__( self, 
                  p_cls_cluster : type = Cluster,
                  p_cluster_limit : int = 0,
                  p_name: str = None, 
                  p_range_max = OATask.C_RANGE_THREAD, 
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
        self._clusters      = {}
        self._cluster_limit = p_cluster_limit

        self._cluster_properties : PropertyDefinitions = self.C_CLUSTER_PROPERTIES.copy()
        self._cluster_properties_dict = {}
        for prop in self._cluster_properties:
            self._cluster_properties_dict[prop[0]] = prop


## -------------------------------------------------------------------------------------------------
    def align_cluster_properties( self, p_properties : PropertyDefinitions ) -> list:
        """
        Aligns list of cluster properties with the given list. In particular, the maximum derivative
        order of numeric properties is aligned. 

        Parameters
        ----------
        p_properties : PropertyDefinitions
            List of properties to be aligned with.

        Returns
        list
            List of unknown properties.
        """

        unknown_properties = []

        for p_ext in p_properties:
            try:
                p_int = self._cluster_properties_dict[p_ext[0]]
                p_int[1] = p_ext[1]   # alignment of maximum order of derivatives
                p_int[2] = p_ext[2]   # alignment of property class
            except:
                # Property not supported by cluster algorithm
                unknown_properties.append(p_ext[0])

        return unknown_properties


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

        return ( self._cluster_limit == 0 ) or ( len(self._clusters.key()) < self._cluster_limit )
    

## -------------------------------------------------------------------------------------------------
    def get_cluster_cls(self):
        return self._cls_cluster
    

## -------------------------------------------------------------------------------------------------
    def get_clusters(self) -> dict[Cluster]:
        """
        This method returns the current list of clusters. 

        Returns
        -------
        dict_of_clusters : dict[Cluster]
            Current dictionary of clusters.
        """

        return self._clusters
    

## -------------------------------------------------------------------------------------------------
    def _add_cluster(self, p_cluster:Cluster) -> bool:
        """
        Protected method to be used to add a new cluster. Please use as part of your algorithm.

        Parameters
        ----------
        p_cluster : Cluster
            Cluster object to be added.

        Returns
        -------
        successful : Bool
            True, if the cluster has been added successfully. False otherwise.
        """

        if not self.new_cluster_allowed(): return False

        self._clusters[p_cluster.get_id()] = p_cluster

        if self.get_visualization(): 
            p_cluster.init_plot( p_figure=self._figure, p_plot_settings=self.get_plot_settings() )

        return True


## -------------------------------------------------------------------------------------------------
    def _remove_cluster(self, p_cluster:Cluster):
        """
        Protected method to remove an existing cluster. Please use as part of your algorithm.

        Parameters
        ----------
        p_cluster : Cluster
            Cluster object to be added.
        """

        p_cluster.remove_plot(p_refresh=True)
        del self._clusters[p_cluster.get_id()]


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

        for cluster in self.get_clusters().values():

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
    def init_plot(self, p_figure: Figure = None, p_plot_settings: PlotSettings = None):

        if not self.get_visualization(): return

        super().init_plot( p_figure=p_figure, p_plot_settings=p_plot_settings)

        for cluster in self._clusters.values():
            cluster.init_plot(p_figure=p_figure, p_plot_settings = p_plot_settings)


## -------------------------------------------------------------------------------------------------
    def update_plot( self, 
                     p_inst_new: List[Instance] = None, 
                     p_inst_del: List[Instance] = None, 
                     **p_kwargs ):

        if not self.get_visualization(): return

        for cluster in self._clusters.values():
            cluster.update_plot(p_inst_new = p_inst_new, p_inst_del = p_inst_del, **p_kwargs)


## -------------------------------------------------------------------------------------------------
    def _renormalize(self, p_normalizer: Normalizer):
        """
        Internal renormalization of all clusters. See method OATask.renormalize_on_event() for further
        information.

        Parameters
        ----------
        p_normalizer : Normalizer
            Normalizer object to be applied on task-specific 
        """

        for cluster in self._clusters.values():
            cluster.renormalize( p_normalizer=p_normalizer )
 