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
## -- 2024-04-22  0.9.0     DA/SK    Class Cluster: general systematics for properties
## -- 2024-04-28  1.0.0     DA       Class Cluster: new parent class Properties
## -- 2024-04-30  1.1.0     DA       Class Cluster: new parent class Renormalizable
## -- 2024-05-02  1.2.0     DA/SK    Class Cluster: first definition of concrete properties
## -- 2024-05-04  1.3.0     DA       Class Cluster: generic property systematics
## -- 2024-05-06  1.4.0     DA       Plot functionality
## -------------------------------------------------------------------------------------------------

"""
Ver. 1.4.0 (2024-05-06)

This module provides templates for clusters to be used in cluster analyzer algorithms.
"""


from mlpro.bf.various import *
from mlpro.bf.math.properties import *
from mlpro.bf.plot import *
from mlpro.bf.streams import *
from mlpro.bf.math.normalizers import Renormalizable





## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class Cluster (Id, Plottable, Properties, Renormalizable):
    """
    Universal template class for a cluster with any number of properties added by a cluster analyzer. 

    Parameters
    ----------
    p_id
        Optional external id.
    p_properties : PropertyDefinitions
        List of property definitions. 
    p_visualize : bool
        Boolean switch for visualisation. Default = False.
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
                  p_properties : PropertyDefinitions = [],
                  p_visualize : bool = False ):

        Id.__init__( self, p_id = p_id )
        Plottable.__init__( self, p_visualize = p_visualize )
        Properties.__init__( self )

        for p in p_properties:
            self.add_property( p_name = p[0], 
                               p_derivative_order_max = p[1], 
                               p_cls = p[2],
                               p_visualize = p_visualize )


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
    def init_plot(self, p_figure: Figure = None, p_plot_settings: PlotSettings = None):

        if not self.get_visualization(): return

        Plottable.init_plot( self, p_figure=p_figure, p_plot_settings=p_plot_settings)

        for prop in self.get_properties().values():
            prop.init_plot(p_figure=p_figure, p_plot_settings = p_plot_settings)


## -------------------------------------------------------------------------------------------------
    def update_plot( self, 
                     p_inst_new: List[Instance] = None, 
                     p_inst_del: List[Instance] = None, 
                     **p_kwargs ):

        if not self.get_visualization(): return

        for prop in self.get_properties().values():
            prop.update_plot(p_inst_new = p_inst_new, p_inst_del = p_inst_del, **p_kwargs)

        Plottable.update_plot( self, p_inst_new = p_inst_new, p_inst_del = p_inst_del, **p_kwargs )