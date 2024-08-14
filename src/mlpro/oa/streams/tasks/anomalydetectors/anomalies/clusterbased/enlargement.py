## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - The integrative middleware framework for standardized machine learning
## -- Package : mlpro.oa.tasks.anomalydetectors.anomalies.clusterbased
## -- Module  : enlargement.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2023-06-08  0.0.0     SK       Creation
## -- 2023-09-12  1.0.0     SK       Release
## -- 2023-11-21  1.0.1     SK       Time Stamp update
## -- 2024-02-25  1.1.0     SK       Visualisation update
## -- 2024-04-10  1.2.0     DA/SK    Refactoring
## -- 2024-05-28  1.3.0     SK       Refactoring
## -------------------------------------------------------------------------------------------------

"""
Ver. 1.3.9 (2024-05-28)

This module provides a template class for cluster enlargement to be used in anomaly detection algorithms.
"""

from mlpro.oa.streams.tasks.anomalydetectors.anomalies.clusterbased.basics import CBAnomaly





## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class ClusterEnlargement (CBAnomaly):
    """
    Event class to be raised when a cluster enlarges.
    
    """

    C_NAME      = 'Cluster Enlargement'


