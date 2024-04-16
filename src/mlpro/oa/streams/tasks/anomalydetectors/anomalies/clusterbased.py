## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - The integrative middleware framework for standardized machine learning
## -- Package : mlpro.oa.tasks.anomalydetectors.anomalies
## -- Module  : clusterbased.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2023-06-08  0.0.0     SK       Creation
## -- 2023-09-12  1.0.0     SK       Release
## -- 2023-11-21  1.0.1     SK       Time Stamp update
## -- 2024-02-25  1.1.0     SK       Visualisation update
## -- 2024-04-10  1.2.0     DA/SK    Refactoring
## -------------------------------------------------------------------------------------------------

"""
Ver. 1.2.0 (2024-04-10)

This module provides templates for anomaly detection to be used in the context of online adaptivity.
"""

from mlpro.oa.streams.basics import Instance, List
from mlpro.oa.streams.tasks.anomalydetectors.anomalies.basics import Anomaly



## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class CBAnomaly (Anomaly):
    """
    Event class to be raised when cluster-based anomalies are detected.
    
    """

    C_NAME      = 'Cluster based Anomaly'

## -------------------------------------------------------------------------------------------------
    def __init__(self,
                 p_instances : Instance = None,
                 p_ano_scores : list = None,
                 p_visualize : bool = False,
                 p_raising_object : object = None,
                 p_det_time : str = None,
                 **p_kwargs):
        
        super().__init__(p_instance=p_instances, p_ano_scores=p_ano_scores,
                         p_visualize=p_visualize, p_raising_object=p_raising_object,
                         p_det_time=p_det_time, **p_kwargs)





## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class ClusterDrift (CBAnomaly):
    """
    Event class to be raised when cluster drift detected.
    
    """

    C_NAME      = 'Cluster drift Anomaly'

## -------------------------------------------------------------------------------------------------
    def __init__(self,
                 p_instances : Instance = None,
                 p_ano_scores : list = None,
                 p_visualize : bool = False,
                 p_raising_object : object = None,
                 p_det_time : str = None,
                 **p_kwargs):
        
        super().__init__(p_instance=p_instances, p_ano_scores=p_ano_scores,
                         p_visualize=p_visualize, p_raising_object=p_raising_object,
                         p_det_time=p_det_time, **p_kwargs)





## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class NewClusterAppearance (CBAnomaly):
    """
    Event class to be raised when a new cluster appears.
    
    """

    C_NAME      = 'New cluster appearance'

## -------------------------------------------------------------------------------------------------
    def __init__(self,
                 p_instances : Instance = None,
                 p_ano_scores : list = None,
                 p_visualize : bool = False,
                 p_raising_object : object = None,
                 p_det_time : str = None,
                 **p_kwargs):
        
        super().__init__(p_instance=p_instances, p_ano_scores=p_ano_scores,
                         p_visualize=p_visualize, p_raising_object=p_raising_object,
                         p_det_time=p_det_time, **p_kwargs)





## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class ClusterDisappearence (CBAnomaly):
    """
    Event class to be raised when a cluster disappears.
    
    """

    C_NAME      = 'Cluster disappearance'

## -------------------------------------------------------------------------------------------------
    def __init__(self,
                 p_instances : Instance = None,
                 p_ano_scores : list = None,
                 p_visualize : bool = False,
                 p_raising_object : object = None,
                 p_det_time : str = None,
                 **p_kwargs):
        
        super().__init__(p_instance=p_instances, p_ano_scores=p_ano_scores,
                         p_visualize=p_visualize, p_raising_object=p_raising_object,
                         p_det_time=p_det_time, **p_kwargs)





## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class ClusterEnlargement (CBAnomaly):
    """
    Event class to be raised when a cluster enlarges.
    
    """

    C_NAME      = 'Cluster Enlargement'

## -------------------------------------------------------------------------------------------------
    def __init__(self,
                 p_instances : Instance = None,
                 p_ano_scores : list = None,
                 p_visualize : bool = False,
                 p_raising_object : object = None,
                 p_det_time : str = None,
                 **p_kwargs):
        
        super().__init__(p_instance=p_instances, p_ano_scores=p_ano_scores,
                         p_visualize=p_visualize, p_raising_object=p_raising_object,
                         p_det_time=p_det_time, **p_kwargs)





## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class ClusterShrinkage (CBAnomaly):
    """
    Event class to be raised when a cluster shrinks.
    
    """

    C_NAME      = 'Cluster shrinkage'

## -------------------------------------------------------------------------------------------------
    def __init__(self,
                 p_instances : Instance = None,
                 p_ano_scores : list = None,
                 p_visualize : bool = False,
                 p_raising_object : object = None,
                 p_det_time : str = None,
                 **p_kwargs):
        
        super().__init__(p_instance=p_instances, p_ano_scores=p_ano_scores,
                         p_visualize=p_visualize, p_raising_object=p_raising_object,
                         p_det_time=p_det_time, **p_kwargs)





## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class ClusterDensityVariation (CBAnomaly):
    """
    Event class to be raised when the density of a cluster changes.
    
    """

    C_NAME      = 'Cluster density variation'

## -------------------------------------------------------------------------------------------------
    def __init__(self,
                 p_instances : Instance = None,
                 p_ano_scores : list = None,
                 p_visualize : bool = False,
                 p_raising_object : object = None,
                 p_det_time : str = None,
                 **p_kwargs):
        
        super().__init__(p_instance=p_instances, p_ano_scores=p_ano_scores,
                         p_visualize=p_visualize, p_raising_object=p_raising_object,
                         p_det_time=p_det_time, **p_kwargs)

