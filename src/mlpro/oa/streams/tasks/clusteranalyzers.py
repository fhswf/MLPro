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
## -- 2023-11-18  0.5.0     DA       Class ClusterCentroid: added plot functionality
## -- 2023-12-08  0.6.0     DA       Class ClusterAnalyzer: 
## --                                - changed internal cluster storage from list to dictionary
## --                                - added method _remove_cluster()
## -- 2023-12-10  0.6.1     DA       Bugfix in method ClusterAnalyzer.get_cluster_membership()
## -- 2023-12-20  0.7.0     DA       Renormalization
## -------------------------------------------------------------------------------------------------

"""
Ver. 0.7.0 (2023-12-20)

This module provides templates for cluster analysis to be used in the context of online adaptivity.
"""

from matplotlib.figure import Figure
from matplotlib.text import Text
from mpl_toolkits.mplot3d.art3d import Line3D, Text3D
from mlpro.bf.mt import Figure, PlotSettings
from mlpro.bf.various import *
from mlpro.bf.plot import *
from mlpro.bf.streams import *
from mlpro.oa.streams import OATask
from mlpro.bf.math.normalizers import Normalizer
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
        self._clusters      = {}
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

        return ( self._cluster_limit == 0 ) or ( len(self._clusters.key()) < self._cluster_limit )
    

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


## -------------------------------------------------------------------------------------------------
    def init_plot(self, p_figure: Figure = None, p_plot_settings: PlotSettings = None, **p_kwargs):
        super().init_plot(p_figure, p_plot_settings, **p_kwargs)
        self._centroid.init_plot( p_figure=p_figure, p_plot_settings=p_plot_settings)


## -------------------------------------------------------------------------------------------------
    def _init_plot_2d(self, p_figure: Figure, p_settings: PlotSettings):
        self._plot_line1 = None
        self._plot_line2 = None
        self._plot_line1_t1 : Text = None
        self._plot_line1_t2 : Text = None
        self._plot_line1_t3 : Text = None
        self._plot_line1_t4 : Text = None
        self._plot_line1_t5 : Text = None
    

## -------------------------------------------------------------------------------------------------
    def _init_plot_3d(self, p_figure: Figure, p_settings: PlotSettings):
        self._plot_line1 : Line3D = None
        self._plot_line2 : Line3D = None
        self._plot_line3 : Line3D = None
        self._plot_line1_t1 : Text3D = None
        self._plot_line1_t2 : Text3D = None
        self._plot_line1_t3 : Text3D = None
        self._plot_line1_t4 : Text3D = None
        self._plot_line1_t5 : Text3D = None
        self._plot_line1_t6 : Text3D = None
        self._plot_line1_t7 : Text3D = None
    

## -------------------------------------------------------------------------------------------------
    def update_plot(self, **p_kwargs):
        super().update_plot(**p_kwargs)
        self._centroid.update_plot( p_kwargs=p_kwargs)   
    

## -------------------------------------------------------------------------------------------------
    def _update_plot_2d(self, p_settings: PlotSettings, **p_kwargs):
        super()._update_plot_2d(p_settings, **p_kwargs)

        # 1 Get coordinates
        centroid = self._centroid.get_values()
        ax_xlim  = p_settings.axes.get_xlim()
        ax_ylim  = p_settings.axes.get_ylim()
        xlim     = [ min( ax_xlim[0], centroid[0] ), max(ax_xlim[1], centroid[0] ) ]
        ylim     = [ min( ax_ylim[0], centroid[1] ), max(ax_ylim[1], centroid[1] ) ]

        # 2 Plot a crosshair
        if self._plot_line1 is None:
            # 2.1 Add initial crosshair lines
            cluster_id = self.get_id()
            col_id = cluster_id % len(self.C_CLUSTER_COLORS)
            color = self.C_CLUSTER_COLORS[col_id]
            label = ' C' + str(cluster_id) + ' '
            self._plot_line1 = p_settings.axes.plot( xlim, [centroid[1],centroid[1]], color=color, linestyle='dashed', lw=1, label=label)[0]
            self._plot_line2 = p_settings.axes.plot( [centroid[0],centroid[0]], ylim, color=color, linestyle='dashed', lw=1)[0]
            self._plot_line1_t1 = p_settings.axes.text(centroid[0], centroid[1], label, color=color )
            self._plot_line1_t2 = p_settings.axes.text(xlim[0], centroid[1], label, ha='right', va='center', color=color )
            self._plot_line1_t3 = p_settings.axes.text(xlim[1], centroid[1], label, ha='left',va='center', color=color )
            self._plot_line1_t4 = p_settings.axes.text(centroid[0], ylim[0], label, ha='center', va='top', color=color )
            self._plot_line1_t5 = p_settings.axes.text(centroid[0], ylim[1], label, ha='center', va='bottom',color=color )
            p_settings.axes.legend(title='Clusters', alignment='left', loc='upper right', shadow=True, draggable=True)
        else:
            # 2.2 Update data of crosshair lines
            self._plot_line1.set_data( xlim, [centroid[1],centroid[1]] )
            self._plot_line2.set_data( [centroid[0],centroid[0]], ylim )
            self._plot_line1_t1.set(position=(centroid[0], centroid[1]) )
            self._plot_line1_t2.set(position=(xlim[0], centroid[1]))
            self._plot_line1_t3.set(position=(xlim[1], centroid[1]))
            self._plot_line1_t4.set(position=(centroid[0], ylim[0]))
            self._plot_line1_t5.set(position=(centroid[0], ylim[1]))


## -------------------------------------------------------------------------------------------------
    def _update_plot_3d(self, p_settings: PlotSettings, **p_kwargs):
        super()._update_plot_3d(p_settings, **p_kwargs) 

        # 1 Get coordinates
        centroid = self._centroid.get_values()
        ax_xlim  = p_settings.axes.get_xlim()
        ax_ylim  = p_settings.axes.get_ylim()
        ax_zlim  = p_settings.axes.get_zlim()
        xlim     = [ min( ax_xlim[0], centroid[0] ), max(ax_xlim[1], centroid[0] ) ]
        ylim     = [ min( ax_ylim[0], centroid[1] ), max(ax_ylim[1], centroid[1] ) ]
        zlim     = [ min( ax_zlim[0], centroid[2] ), max(ax_zlim[1], centroid[2] ) ]


        # 2 Determine label text alignments
        ap = p_settings.axes.get_axis_position()

        if ap[0]: 
            t2_ha='left' 
            t3_ha='right'
        else: 
            t2_ha='right'
            t3_ha='left'

        if ap[1]: 
            t4_ha='right' 
            t5_ha='left'
        else: 
            t4_ha='left'
            t5_ha='right'

        t6_va='top' 
        t7_va='bottom'


        # 3 Plot a crosshair with label texts
        if self._plot_line1 is None:
            # 3.1 Add initial crosshair lines
            cluster_id = self.get_id()
            col_id = cluster_id % len(self.C_CLUSTER_COLORS)
            color = self.C_CLUSTER_COLORS[col_id]
            label = ' C' + str(cluster_id) + ' '
            self._plot_line1 = p_settings.axes.plot( xlim, [centroid[1],centroid[1]], [centroid[2],centroid[2]], color=color, linestyle='dashed', lw=1, label=label)[0]
            self._plot_line2 = p_settings.axes.plot( [centroid[0],centroid[0]], ylim, [centroid[2],centroid[2]], color=color, linestyle='dashed', lw=1)[0]
            self._plot_line3 = p_settings.axes.plot( [centroid[0],centroid[0]], [centroid[1],centroid[1]], zlim, color=color, linestyle='dashed', lw=1)[0]

            self._plot_line1_t1 = p_settings.axes.text(centroid[0], centroid[1], centroid[2], label, color=color )
            self._plot_line1_t2 = p_settings.axes.text(xlim[0], centroid[1], centroid[2], label, ha=t2_ha, va='center', color=color )
            # self._plot_line1_t3 = p_settings.axes.text(xlim[1], centroid[1], centroid[2], label, ha=t3_ha, va='center', color=color )
            self._plot_line1_t4 = p_settings.axes.text(centroid[0], ylim[0], centroid[2], label, ha=t4_ha, va='center', color=color )
            # self._plot_line1_t5 = p_settings.axes.text(centroid[0], ylim[1], centroid[2], label, ha=t5_ha, va='center', color=color )
            self._plot_line1_t6 = p_settings.axes.text(centroid[0], centroid[1], zlim[0], label, ha='center', va=t6_va, color=color )
            # self._plot_line1_t7 = p_settings.axes.text(centroid[0], centroid[1], zlim[1], label, ha='center', va=t7_va, color=color )

            p_settings.axes.legend(title='Clusters', alignment='left', loc='right', shadow=True, draggable=True)
        else:
            # 3.2 Update data of crosshair lines
            self._plot_line1.set_data_3d( xlim, [centroid[1],centroid[1]], [centroid[2],centroid[2]] )
            self._plot_line2.set_data_3d( [centroid[0],centroid[0]], ylim, [centroid[2],centroid[2]] )
            self._plot_line3.set_data_3d( [centroid[0],centroid[0]], [centroid[1],centroid[1]], zlim )

            self._plot_line1_t1.set(position_3d=(centroid[0], centroid[1], centroid[2]))
            self._plot_line1_t2.set(position_3d=(xlim[0], centroid[1], centroid[2]), ha=t2_ha)
            # self._plot_line1_t3.set(position_3d=(xlim[1], centroid[1], centroid[2]), ha=t3_ha)
            self._plot_line1_t4.set(position_3d=(centroid[0], ylim[0], centroid[2]), ha=t4_ha)
            # self._plot_line1_t5.set(position_3d=(centroid[0], ylim[1], centroid[2]), ha=t5_ha)
            self._plot_line1_t6.set(position_3d=(centroid[0], centroid[1], zlim[0]), va=t6_va)
            # self._plot_line1_t7.set(position_3d=(centroid[0], centroid[1], zlim[1]), va=t7_va)


## -------------------------------------------------------------------------------------------------
    def renormalize(self, p_normalizer: Normalizer):
        self._centroid.set_values( p_normalizer.renormalize( self._centroid.get_values() ) )

