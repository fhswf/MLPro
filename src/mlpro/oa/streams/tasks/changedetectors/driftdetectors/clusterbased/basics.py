## -- ----------------------------------------------------------------------------------------------
## -- Project : MLPro - The integrative middleware framework for standardized machine learning
## -- Package : mlpro.oa.streams.tasks.driftdetectors.clusterbased
## -- Module  : basics.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2025-02-12  0.1.0     DA       Creation
## -------------------------------------------------------------------------------------------------

"""
Ver. 0.1.0 (2025-02-12)

This module provides a template for cluster-based drift detection algorithms to be used in the context of online adaptivity.
"""

from mlpro.bf.various import Log
from mlpro.bf.math.properties import PropertyDefinitions

from mlpro.oa.streams.basics import OAStreamTask
from mlpro.oa.streams.tasks.driftdetectors.basics import DriftDetector
from mlpro.oa.streams.tasks.clusteranalyzers.basics import ClusterAnalyzer




## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class DriftDetectorCB (DriftDetector):
    """
    This is the base class for online-adaptive cluster-based drift detectors. It raises an event 
    when a drift is detected in a cluster dataset.

    Parameters
    ----------
    p_clusterer : ClusterAnalyzer
        Related cluster analyzer providing its clusters as the basis for drift detection
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
    """

    C_TYPE = 'Cluster-based Drift Detector'

    # List of cluster properties necessary for the algorithm
    C_REQ_CLUSTER_PROPERTIES : PropertyDefinitions = []

## -------------------------------------------------------------------------------------------------
    def __init__( self,
                  p_clusterer : ClusterAnalyzer,
                  p_name : str = None,
                  p_range_max = OAStreamTask.C_RANGE_THREAD,
                  p_ada : bool = True,
                  p_duplicate_data : bool = False,
                  p_visualize : bool = False,
                  p_logging=Log.C_LOG_ALL,
                  **p_kwargs ):

        super().__init__( p_name = p_name,
                          p_range_max = p_range_max,
                          p_ada = p_ada,
                          p_duplicate_data = p_duplicate_data,
                          p_visualize = p_visualize,
                          p_logging = p_logging,
                          **p_kwargs )
        
        self._clusterer = p_clusterer
        unknown_prop    = self._clusterer.align_cluster_properties(p_properties=self.C_REQ_CLUSTER_PROPERTIES)

        if len(unknown_prop) > 0:
           raise RuntimeError("The following cluster properties need to be provided by the clusterer: ", unknown_prop)
        