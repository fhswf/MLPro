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
## -- 2024-05-27  1.0.0     DA       Initial design finished
## -- 2024-05-28  1.0.1     DA       Bugfix in ClusterAnalyzer.new_cluster_allowed()
## -- 2024-06-05  1.0.2     DA       Bugfix in ClusterAnalyzer.get_cluster_membership()
## -- 2024-06-06  1.1.0     DA       New method ClusterAnalyzer._get_next_cluster_id()
## -- 2024-06-08  1.2.0     DA       Refactoring class ClusterAnalyzer: 
## --                                - renamed attributes C_MS_SCOPE_* to C_RESULT_SCOPE_*
## --                                - new method _get_cluster_relations()
## --                                - new method get_cluster_influences()
## -- 2024-06-16  1.2.1     DA       Bugfix in ClusterAnalyzer.align_cluster_properties()
## -- 2024-08-20  1.3.0     DA       Raising of events Cluster.C_CLUSTER_ADDED, Cluster.C_CLUSTER_REMOVED
## -- 2024-08-21  1.3.1     DA       Resolved name collision of class mlpro.bf.events.Event
## -- 2025-04-13  1.4.0     DA       Refactoring of ClusterAnalyzer:
## --                                - provision of current clusters as public attribute clusters
## --                                - removed the get_clusters() method
## --                                - renamed the _get_next_cell_id() method to _get_next_cluster_id()
## -- 2025-04-24  1.5.0     DA       Added method _get_clusters() since needed for wrappers(!!)
## -- 2025-04-27  1.5.1     DA       Class ClusterAnalyzer: changed internal access to clusters to 
## --                                self._clusters 
## -- 2025-06-06  1.6.0     DA       Refactoring: p_inst -> p_instances
## -------------------------------------------------------------------------------------------------

"""
Ver. 1.6.0 (2025-06-06)

This module provides a template class for online cluster analysis.
"""


from typing import List, Tuple

try:
    from matplotlib.figure import Figure
except:
    class Figure : pass

from mlpro.bf.events import Event as MLProEvent
from mlpro.bf.math.properties import *
from mlpro.bf.mt import PlotSettings
from mlpro.bf.streams import Instance, InstDict
from mlpro.bf.various import *
from mlpro.bf.plot import *
from mlpro.oa.streams import OAStreamTask
from mlpro.bf.math.normalizers import Normalizer
from mlpro.oa.streams.tasks.clusteranalyzers.clusters import Cluster, ClusterId


# Export list for public API
__all__ = [ 'ClusterAnalyzer',
            'ClusterId',
            'ResultItem' ]



ResultItem = Tuple[ClusterId, float, object]



## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class ClusterAnalyzer (OAStreamTask):
    """
    Base class for online cluster analysis. It raises an event when a cluster was added or removed.

    Steps to implement a new algorithm are:
    - Create a new class and inherit from this base class
    - Specify all cluster properties provided/maintained by your algorithm in C_CLUSTER_PROPERTIES.
    - Implement method self._adapt() to update your cluster list on new instances
    - Implement method self._adapt_reverse() to update your cluster list on obsolete instances
    - New cluster: hand over self._cluster_properties.values() on instantiation
    
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
    C_RESULT_SCOPE_ALL : int = 0
        Result scope, that includes all clusters
    C_RESULT_SCOPE_NONZERO : int = 1
        Result scope, that includes just clusters with result values > 0
    C_RESULT_SCOPE_MAX : int = 2
        Result scope, that includes just the cluster with the highest result value.
    C_CLUSTER_PROPERTIES : PropertyDefinitions
        List of cluster properties supported/maintained by the algorithm. These properties 
        are handed over to each new cluster.
    """

    C_TYPE                          = 'Cluster Analyzer'

    C_EVENT_CLUSTER_ADDED           = 'CLUSTER_ADDED'
    C_EVENT_CLUSTER_REMOVED         = 'CLUSTER_REMOVED'

    C_PLOT_ACTIVE                   = True
    C_PLOT_STANDALONE               = False

    # Possible result scopes for methods get_cluster_memberships() and get_cluster_influences()
    C_RESULT_SCOPE_ALL : int        = 0
    C_RESULT_SCOPE_NONZERO : int    = 1
    C_RESULT_SCOPE_MAX : int        = 2

    # List of cluster properties supported/maintained by the algorithm
    C_CLUSTER_PROPERTIES : PropertyDefinitions = []

## -------------------------------------------------------------------------------------------------
    def __init__( self, 
                  p_cls_cluster : type = Cluster,
                  p_cluster_limit : int = 0,
                  p_name: str = None, 
                  p_range_max = OAStreamTask.C_RANGE_THREAD, 
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
        self._next_cluster_id : ClusterId = -1

        self._cluster_properties = {}
        for prop in self.C_CLUSTER_PROPERTIES:
            self._cluster_properties[prop[0]] = prop


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
                p_int = self._cluster_properties[p_ext[0]]

                # If the property is basically provided it is aligned with external settings
                self._cluster_properties[p_ext[0]] = p_ext
            except:
                # Property not supported by cluster algorithm
                unknown_properties.append(p_ext[0])

        return unknown_properties


## -------------------------------------------------------------------------------------------------
    def _run(self, p_instances : InstDict):
        self.adapt( p_instances = p_instances )


## -------------------------------------------------------------------------------------------------
    def new_cluster_allowed(self) -> bool:
        """
        Determines whether adding a new cluster is allowed.

        Returns
        -------
        bool
           True, if adding a new cluster allowed. False otherwise.
        """

        return ( self._cluster_limit == 0 ) or ( len(self._clusters.keys()) < self._cluster_limit )
    

## -------------------------------------------------------------------------------------------------
    def get_cluster_cls(self):
        return self._cls_cluster
    

## -------------------------------------------------------------------------------------------------
    def _get_clusters(self):
        return self._clusters


## -------------------------------------------------------------------------------------------------
    def _get_next_cluster_id(self) -> ClusterId:
        self._next_cluster_id += 1
        return self._next_cluster_id
    

## -------------------------------------------------------------------------------------------------
    def _add_cluster(self, p_cluster:Cluster) -> bool:
        """
        Protected method to be used to add a new cluster. Please use as part of your algorithm. 
        Please use method new_cluster_allowed() before adding a cluster.

        Parameters
        ----------
        p_cluster : Cluster
            Cluster object to be added.
        """

        self._clusters[p_cluster.id] = p_cluster

        if self.get_visualization(): 
            p_cluster.init_plot( p_figure=self._figure, p_plot_settings=self.get_plot_settings() )

        self._raise_event( p_event_id = self.C_EVENT_CLUSTER_ADDED, 
                           p_event_object = MLProEvent( p_raising_object = self,
                                                        p_cluster = p_cluster ) )


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
        del self._clusters[p_cluster.id]

        self._raise_event( p_event_id = self.C_EVENT_CLUSTER_REMOVED, 
                           p_event_object = MLProEvent( p_raising_object = self,
                                                        p_cluster = p_cluster ) )


## -------------------------------------------------------------------------------------------------
    def _get_cluster_relations( self, 
                                p_relation_type : int,
                                p_instance : Instance,
                                p_scope : int ) -> List[ResultItem]:
        """
        Internal method to determine the relation of the given instance to each cluster as a value in 
        percent. Currently supported relations are membership and influence. 

        See also: public methods get_cluster_memberships() and get_cluster influences()

                
        Parameters
        ----------
        p_relation_type : int
            Possible values are 0 (cluster membership) and 1 (cluster influence)
        p_instance : Instance
            Instance to be evaluated.
        p_scope : int
            Scope of the result list. See class attributes C_RESULT_SCOPE_* for possible values.

        Returns
        -------
        results : List[ResultItem]
            List of result items which are tuples of a cluster id, a relative result 
            value in [0,1] and a reference to the cluster object.
        """

        # 1 Determination of membership values of the instance for all clusters
        sum_results         = 0
        list_results_abs    = []
        list_results_rel    = []
        cluster_max_results = None

        for cluster in self._clusters.values():

            if p_relation_type == 0:
                result_abs  = cluster.get_membership( p_instance = p_instance )
            else:
                result_abs  = cluster.get_influence( p_instance = p_instance )

            sum_results += result_abs

            if ( p_scope != self.C_RESULT_SCOPE_ALL ) and ( result_abs == 0 ): continue

            if p_scope == self.C_RESULT_SCOPE_MAX:
                # Cluster with highest membership value is buffered
                if ( cluster_max_results is None ) or ( result_abs > cluster_max_results[1] ):
                    cluster_max_results = ( cluster, result_abs )
            else:
                list_results_abs.append( (cluster, result_abs) )

        if cluster_max_results is not None:
            list_results_abs.append( cluster_max_results )            
            sum_results = cluster_max_results[1]


        # 2 Determination of relative result values according to the required scope
        for result_abs in list_results_abs:
            try:
                result_rel = result_abs[1] / sum_results
            except:
                result_rel = 0

            list_results_rel.append( ( result_abs[0].id, result_rel, result_abs[0] ) )

        return list_results_rel

    
## -------------------------------------------------------------------------------------------------
    def get_cluster_memberships( self, 
                                 p_instance : Instance,
                                 p_scope : int = C_RESULT_SCOPE_MAX ) -> List[ResultItem]:
        """
        Method to determine the relative membership of the given instance to each cluster as a value 
        in [0,1]. 
        
        See also: method Cluster.get_membership().

        Parameters
        ----------
        p_instance : Instance
            Instance to be evaluated.
        p_scope : int
            Scope of the result list. See class attributes C_RESULT_SCOPE_* for possible values. Default
            value is C_RESULT_SCOPE_MAX.

        Returns
        -------
        List[ResultItem]
            List of membership items which are tuples of a cluster id, a relative membership value 
            in [0,1], and a reference to the cluster object.
        """

        return self._get_cluster_relations( p_relation_type = 0,
                                            p_instance = p_instance,
                                            p_scope = p_scope )
    

## -------------------------------------------------------------------------------------------------
    def get_cluster_influences( self, 
                                p_instance : Instance,
                                p_scope : int = C_RESULT_SCOPE_MAX ) -> List[ResultItem]:
        """
        Method to determine the relative influence of the given instance to each cluster as a value 
        in [0,1]. 
        
        See also: method Cluster.get_influence().

        Parameters
        ----------
        p_instance : Instance
            Instance to be evaluated.
        p_scope : int
            Scope of the result list. See class attributes C_RESULT_SCOPE_* for possible values. Default
            value is C_RESULT_SCOPE_MAX.

        Returns
        -------
        List[ResultItem]
            List of influence items which are tuples of a cluster id, a relative influence value in 
            [0,1], and a reference to the cluster object.
        """

        return self._get_cluster_relations( p_relation_type = 1,
                                            p_instance = p_instance,
                                            p_scope = p_scope )

        
## -------------------------------------------------------------------------------------------------
    def init_plot(self, p_figure: Figure = None, p_plot_settings: PlotSettings = None):

        if not self.get_visualization(): return

        super().init_plot( p_figure=p_figure, p_plot_settings=p_plot_settings)

        for cluster in self._clusters.values():
            cluster.init_plot(p_figure=p_figure, p_plot_settings = p_plot_settings)


## -------------------------------------------------------------------------------------------------
    def update_plot( self, 
                     p_instances : InstDict = None, 
                     **p_kwargs ):

        if not self.get_visualization(): return

        for cluster in self._clusters.values():
            cluster.update_plot( p_instances = p_instances, **p_kwargs)


## -------------------------------------------------------------------------------------------------
    def remove_plot(self, p_refresh:bool = True):
        """"
        Removes the plot and optionally refreshes the display.

        Parameters
        ----------
        p_refresh : bool = True
            On True the display is refreshed after removal
        """

        if not self.get_visualization(): return

        for cluster in self._clusters.values():
            cluster.remove_plot( p_refresh = False)

        
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
 

## -------------------------------------------------------------------------------------------------
    clusters = property( fget = _get_clusters )