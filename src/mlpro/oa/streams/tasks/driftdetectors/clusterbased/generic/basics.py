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
## -------------------------------------------------------------------------------------------------

"""
Ver. 0.3.0 (2025-04-01)

This module provides template classes for generic cluster-based drift detection
"""

from mlpro.bf.various import Log
from mlpro.bf.math.properties import *
from mlpro.bf.streams import InstDict, InstTypeNew
from mlpro.oa.streams import OAStreamTask
from mlpro.oa.streams.tasks.clusteranalyzers import ClusterAnalyzer, Cluster
from mlpro.oa.streams.tasks.driftdetectors.clusterbased.basics import DriftDetectorCB



## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class DriftDetectorCBGeneric ( DriftDetectorCB ):
    """
    Template for generic cluster-based drift detectors observing multiple properties.

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
                  p_cls_drift : type,
                  p_name : str = None,
                  p_range_max = OAStreamTask.C_RANGE_THREAD,
                  p_ada : bool = True,
                  p_duplicate_data : bool = False,
                  p_visualize : bool = False,
                  p_logging=Log.C_LOG_ALL,
                  p_drift_buffer_size : int = 100,
                  **p_kwargs ):
        
        self.C_REQ_CLUSTER_PROPERTIES = p_properties
        self._cls_drift               = p_cls_drift
        self.cluster_drifts           = {}

        super().__init__( p_clusterer = p_clusterer,
                          p_name = p_name,
                          p_range_max = p_range_max,
                          p_ada = p_ada,
                          p_duplicate_data = p_duplicate_data,
                          p_visualize = p_visualize,
                          p_logging= p_logging,
                          p_drift_buffer_size = p_drift_buffer_size,
                          **p_kwargs )
        

## -------------------------------------------------------------------------------------------------
    def _buffer_drift(self, p_drift):

        # 0 Buffer the drift event
        super()._buffer_drift(p_drift)


        # 1 Get the id of the first cluster stored within the drift object
        cluster_id = next( iter(p_drift.clusters.keys()) )


        # 2 Store this drift event for the related cluster
        self.cluster_drifts[cluster_id] = p_drift
 

## -------------------------------------------------------------------------------------------------
    def _run(self, p_inst : InstDict ):

        # 1 Get current list of clusters
        clusters = self._clusterer.get_clusters()


        # 2 Observation of clusters
        for cluster in clusters.values():

            # 2.1 Determine the current drift status
            drift_status = self._get_drift_status( p_cluster = cluster,
                                                   p_properties = self.C_REQ_CLUSTER_PROPERTIES,
                                                   **self.kwargs )
            
            # 2.2 Get last drift event for the current cluster from the internal buffer
            try:
                existing_drift = self.cluster_drifts[cluster.id]
            except:
                existing_drift = None


            # 2.3 Raise a new drift event, whenever a drift of this cluster is determined for the first time
            #     or if the drift status changes.
            if ( ( existing_drift is None ) and ( drift_status == True ) ) or \
               ( ( existing_drift is not None ) and ( existing_drift.drift_status != drift_status ) ):
                new_drift = self._cls_drift( p_drift_status = drift_status,
                                             p_tstamp = self._get_tstamp(),
                                             p_visualize = self.get_visualization(),
                                             p_raising_object = self,
                                             p_clusters = { cluster.id : cluster },
                                             p_properties = self.C_REQ_CLUSTER_PROPERTIES,
                                             **self.kwargs )
               
                self._raise_drift_event( p_drift = new_drift )
              

        # 3 Update of stored last drift events per cluster
        for cluster_id, drift in list( self.cluster_drifts.items() ):
                        
            try:
                # 3.1 Check whether the related cluster still exists
                related_cluster = clusters[cluster_id]
            except:
                # 3.2 Cluster disappered
                if drift.drift_status == True:
                    new_drift = self._cls_drift( p_drift_status = False,
                                                 p_tstamp = self._get_tstamp(),
                                                 p_visualize = self.get_visualization(),
                                                 p_raising_object = self,
                                                 p_clusters = { cluster_id : cluster },
                                                 p_properties = self.C_REQ_CLUSTER_PROPERTIES,
                                                 **self.kwargs )
                
                    self._raise_drift_event( p_drift = new_drift ) 

                del self.cluster_drifts[cluster_id]   
                       

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





## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class DriftDetectorCBGenSingle ( DriftDetectorCBGeneric ):
    """
    Specialized template for generic cluster-based drift detectors observing a single property.

    Parameters
    ----------
    ...
    p_property : PropertyDefinition
        Single property to be observed.
    ...
    """

    C_TYPE = 'Cluster-based Generic Drift Detector (Single)'

## -------------------------------------------------------------------------------------------------
    def __init__( self,
                  p_clusterer : ClusterAnalyzer,
                  p_property : PropertyDefinition,
                  p_cls_drift : type,
                  p_name : str = None,
                  p_range_max = OAStreamTask.C_RANGE_THREAD,
                  p_ada : bool = True,
                  p_duplicate_data : bool = False,
                  p_visualize : bool = False,
                  p_logging = Log.C_LOG_ALL,
                  **p_kwargs ):
        
        super().__init__( p_clusterer = p_clusterer,
                          p_properties = [ p_property ],
                          p_cls_drift = p_cls_drift,
                          p_name = p_name,
                          p_range_max = p_range_max,
                          p_ada = p_ada,
                          p_duplicate_data = p_duplicate_data,
                          p_visualize = p_visualize,
                          p_logging = p_logging,
                          **p_kwargs )





## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class DriftDetectorCBGenMulti ( DriftDetectorCBGeneric ):
    """
    Specialized template for generic cluster-based drift detectors observing a single property.
    """

    C_TYPE = 'Cluster-based Generic Drift Detector (Multi)'