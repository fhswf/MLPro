## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - The integrative middleware framework for standardized machine learning
## -- Package : mlpro.oa.tasks.driftdetectors.clusterbased.generic
## -- Module  : basics.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2025-03-04  0.1.0     DA/DS    Creation
## -- 2025-03-18  0.2.0     DA/DS    Completion of method DriftDetectorCBGeneric._run()
## -- 2025-03-26  0.2.1     DA       Bugfix in method DriftDetectorCBGeneric._run()
## -- 2025-04-01  0.3.0     DA       Class DriftDetectorCBGeneric: integration of new method 
## --                                _get_tstamp()
## -- 2025-04-13  0.4.0     DA       Refactoring
## -- 2025-06-10  0.5.0     DA/DS    Refactoring
## -------------------------------------------------------------------------------------------------

"""
Ver. 0.5.0 (2025-06-10)

This module provides template classes for generic cluster-based drift detection
"""

from mlpro.bf.various import Log
from mlpro.bf.math.properties import PropertyDefinitions
from mlpro.bf.streams import Instance

from mlpro.oa.streams import OAStreamTask
from mlpro.oa.streams.tasks.clusteranalyzers import ClusterAnalyzer, Cluster
from mlpro.oa.streams.tasks.changedetectors.driftdetectors.drifts import Drift
from mlpro.oa.streams.tasks.changedetectors.driftdetectors.clusterbased import DriftDetectorCB
from mlpro.oa.streams.tasks.changedetectors.driftdetectors.drifts.clusterbased  import DriftCB



## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class DriftDetectorCBGeneric ( DriftDetectorCB ):
    """
    Template for generic cluster-based drift detectors for single cluster drifts.

    Parameters
    ----------
    ...
    p_property : PropertyDefinition
        Cluster property to be observed.
    p_cls_drift : type
        Type of drift events to be raised.
    ...
    """

    C_TYPE = 'Cluster-based Generic Drift Detector'

## -------------------------------------------------------------------------------------------------
    def __init__( self,
                  p_clusterer : ClusterAnalyzer,
                  p_properties : PropertyDefinitions,
                  p_cls_drift : type[DriftCB],
                  p_name : str = None,
                  p_range_max = OAStreamTask.C_RANGE_THREAD,
                  p_ada : bool = True,
                  p_duplicate_data : bool = False,
                  p_visualize : bool = False,
                  p_logging=Log.C_LOG_ALL,
                  p_drift_buffer_size : int = 100,
                  p_thrs_inst : int = 0,
                  p_thrs_clusters : int = 1,
                  **p_kwargs ):
        
        self.C_REQ_CLUSTER_PROPERTIES   = p_properties
        self._cls_drift : type[DriftCB] = p_cls_drift
        self.cluster_drifts             = {}

        super().__init__( p_clusterer = p_clusterer,
                          p_property= p_properties,
                          p_name = p_name,
                          p_range_max = p_range_max,
                          p_ada = p_ada,
                          p_duplicate_data = p_duplicate_data,
                          p_visualize = p_visualize,
                          p_logging= p_logging,
                          p_drift_buffer_size = p_drift_buffer_size,
                          p_thrs_inst = p_thrs_inst,
                          p_thrs_clusters = p_thrs_clusters,
                          **p_kwargs )
        

## -------------------------------------------------------------------------------------------------
    def _buffer_drift(self, p_drift : Drift):
        super()._buffer_drift(p_drift)
        cluster_id = next( iter(p_drift.clusters.keys()) )
        self.cluster_drifts[cluster_id] = p_drift


## -------------------------------------------------------------------------------------------------
    def _remove_drift(self, p_drift : Drift):
        super()._remove_drift(p_drift)
        cluster_id = next( iter(p_drift.clusters.keys()) )
        del self.cluster_drifts[cluster_id]
 

## -------------------------------------------------------------------------------------------------
    def _detect(self, p_clusters : dict, p_instance : Instance, **p_kwargs):
        
        # 1 Observation of clusters
        for cluster in p_clusters.values():

            # 1.1 Determine the current drift status
            drift_status = self._get_drift_status( p_cluster = cluster,
                                                   p_properties = self.C_REQ_CLUSTER_PROPERTIES,
                                                   **self.kwargs )
            
            # 1.2 Get last drift event for the current cluster from the internal buffer
            try:
                existing_drift = self.cluster_drifts[cluster.id]
            except:
                existing_drift = None


            # 1.3 Determine whether a new drift event on/off is to be raised
            drift : DriftCB = None

            if ( existing_drift is None ) and ( drift_status == True ):
                # 1.3.1 A new drift is detected
                drift = self._cls_drift( p_status = True,
                                         p_tstamp = p_instance.tstamp,
                                         p_visualize = self.get_visualization(),
                                         p_raising_object = self,
                                         p_clusters = { cluster.id : cluster },
                                         p_properties = self.C_REQ_CLUSTER_PROPERTIES,
                                         **self.kwargs )
                
            elif ( existing_drift is not None ) and ( existing_drift.status != drift_status ):
                # 1.3.2 An existing drift changed its status
                drift = existing_drift
                drift.tstamp = p_instance.tstamp
                
                
            if drift: self._raise_drift_event( p_drift = drift )
                       

## -------------------------------------------------------------------------------------------------
    def _get_drift_status( self, p_cluster : Cluster, p_properties : PropertyDefinitions, **p_kwargs ) -> bool:
        """
        Custom method to determine the drift status of the given cluster.

        Parameters
        ----------
        p_cluster : Cluster
            Cluster to be observed.
        p_properties : PropertyDefinitions
            List of cluster properties to be processed.
        **p_kwargs
            Dictionary with further keyword arguments parameterizing the detection.

        Returns
        -------
        bool
            True, if the cluster is drifting. False otherwise.
        """

        raise NotImplementedError
