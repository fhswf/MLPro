## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - The integrative middleware framework for standardized machine learning
## -- Package : mlpro.oa.streams.tasks.classifiers
## -- Module  : basics.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2024-05-22  0.0.0     DA       Creation
## -------------------------------------------------------------------------------------------------

"""
Ver. 0.0.0 (2024-05-22)

This module provides templates for classifiers to be used in the context of online adaptivity.

"""

from mlpro.oa.streams import OATask




## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class Classifier (OATask):
    """
    Base class for online classification. 

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

    C_TYPE                  = 'Classifier'

    C_PLOT_ACTIVE           = False
    C_PLOT_STANDALONE       = False


