## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - The integrative middleware framework for standardized machine learning
## -- Package : mlpro.oa.streams.tasks.classification.classes
## -- Module  : basics.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2024-04-14  0.0.0     DA       Creation
## -- 2024-04-14  0.1.0     DA       First implementation of classes Class
## -------------------------------------------------------------------------------------------------

"""
Ver. 0.8.3 (2024-04-10)

This module provides templates for classes to be used in classification algorithms.
"""

from mlpro.bf.mt import Figure, PlotSettings
from mlpro.bf.various import *
from mlpro.bf.plot import *
from mlpro.bf.streams import *
from mlpro.bf.math.normalizers import Normalizer




## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class Class (Id, Plottable):
    """
    Base class for a class. 

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
