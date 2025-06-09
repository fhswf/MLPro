## -- ----------------------------------------------------------------------------------------------
## -- Project : MLPro - The integrative middleware framework for standardized machine learning
## -- Package : mlpro.oa.streams.tasks.driftdetectors.clusterbased
## -- Module  : basics.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2025-02-12  0.1.0     DA       Creation
## -- 2025-05-05  0.2.0     Ds       Refactoring : DriftDetectorCBSingle, DriftDetectorCBMulti
## -- 2025-06-09  1.0.0     DA       Refactoring: new parent ChangeDetectorCB
## -------------------------------------------------------------------------------------------------

"""
Ver. 1.0.0 (2025-06-09)

This module provides a template for cluster-based drift detection algorithms.
"""

from mlpro.bf.various import Log
from mlpro.bf.math.properties import *

from mlpro.oa.streams.tasks.changedetectors import Change
from mlpro.oa.streams.tasks.changedetectors.clusterbased import ChangeDetectorCB
from mlpro.oa.streams.tasks.changedetectors.driftdetectors.basics import DriftDetector
from mlpro.oa.streams.tasks.changedetectors.driftdetectors.drifts.clusterbased import *
from mlpro.oa.streams.tasks.clusteranalyzers.basics import ClusterAnalyzer




## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class DriftDetectorCB (ChangeDetectorCB, DriftDetector):
    """
    Base class for cluster-based drift detectors.

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
    p_drift_buffer_size : int = 100
        Size of the internal drift buffer self.anomalies. Default = 100.
    p_thrs_inst : int = 0
        The algorithm is only executed after this number of instances.
    p_kwargs : dict
        Further optional named parameters.
    """

    C_TYPE = 'Cluster-based Drift Detector'

## -------------------------------------------------------------------------------------------------
    def __init__( self,
                  p_clusterer : ClusterAnalyzer,
                  p_name:str = None,
                  p_range_max = DriftDetector.C_RANGE_THREAD,
                  p_ada : bool = True,
                  p_duplicate_data : bool = False,
                  p_visualize : bool = False,
                  p_logging=Log.C_LOG_ALL,
                  p_drift_buffer_size : int = 100,
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
                                   p_change_buffer_size = p_drift_buffer_size,
                                   p_thrs_inst = p_thrs_inst,
                                   p_thrs_clusters = p_thrs_clusters,
                                   **p_kwargs )
        
        DriftDetector.__init__( p_name = p_name,
                                  p_range_max = p_range_max,
                                  p_ada = p_ada,
                                  p_duplicate_data = p_duplicate_data,
                                  p_visualize = p_visualize,
                                  p_logging = p_logging,
                                  p_drift_buffer_size = p_drift_buffer_size,
                                  p_thrs_inst = p_thrs_inst,
                                  **p_kwargs )
        

## -------------------------------------------------------------------------------------------------
    def _triage(self, p_change : Change, **p_kwargs) -> bool:
        return DriftDetector()._triage(self, p_change = p_change, **p_kwargs)