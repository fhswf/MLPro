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
## -- 2024-05-22  1.5.0     DA       Refactoring
## -- 2024-05-25  1.6.0     DA       Aliases ClusterId, MembershipValue
## -- 2024-05-27  1.7.0     DA       Refactoring
## -- 2024-05-29  1.8.0     DA       Class Cluster: order of colors changed
## -- 2024-06-08  1.9.0     DA       New method Cluster.get_influence()
## -- 2024-06-18  2.0.0     DA       Class Cluster: new parent class KWArgs
## -- 2024-07-08  2.1.0     DA       Class Cluster: hand over of kwargs to inner properties
## -- 2025-06-06  2.2.0     DA       Refactoring: p_inst -> p_instances
## -- 2025-06-11  2.3.0     DA       New method Cluster.update_properties()
## -------------------------------------------------------------------------------------------------

"""
Ver. 2.3.0 (2025-06-11)

This module provides a template class for clusters to be used in cluster analyzer algorithms.

"""


from mlpro.bf.various import Id, KWArgs, TStampType
from mlpro.bf.plot import PlotSettings
from mlpro.bf.math.properties import PropertyDefinitions, Properties
from mlpro.bf.streams import Instance



# Export list for public API
__all__ = [ 'Cluster',
            'ClusterId' ]




ClusterId = int



## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class Cluster (Id, Properties, KWArgs):
    """
    Universal template class for a cluster with any number of properties added by a cluster analyzer. 

    Parameters
    ----------
    p_id : ClusterId
        Unique cluster id.
    p_properties : PropertyDefinitions
        List of property definitions. 
    p_visualize : bool
        Boolean switch for visualisation. Default = False.
    p_kwargs : dict
        Further parameters.
    """

    C_PLOT_ACTIVE           = True
    C_PLOT_STANDALONE       = False
    C_PLOT_VALID_VIEWS      = [ PlotSettings.C_VIEW_2D, 
                                PlotSettings.C_VIEW_3D, 
                                PlotSettings.C_VIEW_ND ]
    C_PLOT_DEFAULT_VIEW     = PlotSettings.C_VIEW_ND

    C_CLUSTER_COLORS        = [ 'brown', 
                                'olive', 
                                'orange', 
                                'green', 
                                'red', 
                                'gray', 
                                'purple', 
                                'pink', 
                                'cyan', 
                                'blue' ]

## -------------------------------------------------------------------------------------------------
    def __init__( self, 
                  p_id : ClusterId,
                  p_properties : PropertyDefinitions = [],
                  p_visualize : bool = False,
                  **p_kwargs ):

        KWArgs.__init__( self, **p_kwargs )
        Properties.__init__( self, p_properties = p_properties, p_visualize = p_visualize, **p_kwargs )
        Id.__init__( self, p_id = p_id )


## -------------------------------------------------------------------------------------------------
    def set_plot_color(self, p_color):
        Properties.set_plot_color( self, p_color = p_color)
        

## -------------------------------------------------------------------------------------------------
    def get_membership(self, p_instance : Instance ) -> float:
        """
        Custom method to determine a scalar membership value for the given instance.

        Parameters
        ----------
        p_instance : Instance
            Instance to be examined for membership.

        Returns
        -------
        float
            A scalar value in [0,1] that determines the given instance's membership in this cluster. 
            A value of 0 means that the given instance is not a member of the cluster at all while
            a value of 1 confirms full membership.
        """

        raise NotImplementedError
    

## -------------------------------------------------------------------------------------------------
    def get_influence(self, p_instance : Instance ) -> float:
        """
        Custom method to compute a scalar influence value for the given instance.

        Parameters
        ----------
        p_instance : Instance
            Instance to be examined for its influence to the cluster.

        Returns
        -------
        float
            Scalar value >= 0 that determines the influence of the cluster on the specified instance. 
            A value 0 means that the cluster has no influence on the instance at all.
        """

        raise NotImplementedError
    

## -------------------------------------------------------------------------------------------------
    def update_properties(self, p_tstamp : TStampType ):
        """
        Custom method to update inner cluster properties. To be triggered by the cluster analyzer.

        Parameters
        ----------
        p_tstamp : TStampType
            Time stamp of property update.
        """

        pass


## -------------------------------------------------------------------------------------------------
    color = property( fget = Properties.get_plot_color, fset = set_plot_color )