## -- ----------------------------------------------------------------------------------------------
## -- Project : MLPro - The integrative middleware framework for standardized machine learning
## -- Package : mlpro.oa.tasks.anomalydetectors.cb_detectors
## -- Module  : basics.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2023-06-08  0.0.0     SK       Creation
## -- 2023-09-12  1.0.0     SK       Release
## -- 2023-11-21  1.0.1     SK       Time Stamp update
## -- 2024-02-25  1.1.0     SK       Visualisation update
## -- 2024-04-10  1.2.0     DA/SK    Refactoring
## -- 2024-05-28  1.3.0     SK       Refactoring
## -- 2025-06-04  2.0.0     DA       Refactoring: new parent ChangeDetectorCB
## -------------------------------------------------------------------------------------------------

"""
Ver. 2.0.0 (2025-06-04)

This module provides templates for cluster-based anomaly detection algorithms.
"""

from mlpro.bf.various import Log
from mlpro.bf.math.properties import *

from mlpro.oa.streams.basics import StreamTask, InstDict, InstTypeNew
from mlpro.oa.streams.tasks.changedetectors.clusterbased import ChangeDetectorCB
from mlpro.oa.streams.tasks.changedetectors.anomalydetectors.basics import AnomalyDetector
from mlpro.oa.streams.tasks.changedetectors.anomalydetectors.anomalies.clusterbased import *
from mlpro.oa.streams.tasks.clusteranalyzers.basics import ClusterAnalyzer




## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class AnomalyDetectorCB (ChangeDetectorCB, AnomalyDetector):
    """
    Base class for cluster-based anomaly detectors.

    Parameters
    ----------
    p_clusterer : ClusterAnalyzer
        Associated cluster analyzer.
    p_name : str = None
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
    p_anomaly_buffer_size : int = 100
        Size of the internal anomaly buffer self.anomalies. Default = 100.
    p_thrs_inst : int = 0
        The algorithm is only executed after this number of instances.
    p_kwargs : dict
        Further optional named parameters.
    """

    C_TYPE = 'Cluster-based Anomaly Detector'

## -------------------------------------------------------------------------------------------------
    def __init__( self,
                  p_clusterer : ClusterAnalyzer,
                  p_name:str = None,
                  p_range_max = AnomalyDetector.C_RANGE_THREAD,
                  p_ada : bool = True,
                  p_duplicate_data : bool = False,
                  p_visualize : bool = False,
                  p_logging=Log.C_LOG_ALL,
                  p_anomaly_buffer_size : int = 100,
                  p_thrs_inst : int = 0,
                  p_thrs_clusters : int = 1,
                  **p_kwargs ):
   
        ChangeDetectorCB.__init__( p_clusterer = p_clusterer,
                                   p_name = p_name,
                                   p_range_max = p_range_max,
                                   p_ada = p_ada,
                                   p_duplicate_data = p_duplicate_data,
                                   p_visualize = p_visualize,
                                   p_logging = p_logging,
                                   p_change_buffer_size = p_anomaly_buffer_size,
                                   p_thrs_inst = p_thrs_inst,
                                   p_thrs_clusters = p_thrs_clusters,
                                   **p_kwargs )
        
        AnomalyDetector.__init__( p_name = p_name,
                                  p_range_max = p_range_max,
                                  p_ada = p_ada,
                                  p_duplicate_data = p_duplicate_data,
                                  p_visualize = p_visualize,
                                  p_logging = p_logging,
                                  p_anomaly_buffer_size = p_anomaly_buffer_size,
                                  p_thrs_inst = p_thrs_inst,
                                  **p_kwargs )
        

## -------------------------------------------------------------------------------------------------
    def _triage(self, p_change, **p_kwargs):
        return AnomalyDetector()._triage(self, p_change = p_change, **p_kwargs)
        

