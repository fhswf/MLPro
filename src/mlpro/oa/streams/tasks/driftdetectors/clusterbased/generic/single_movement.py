## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - The integrative middleware framework for standardized machine learning
## -- Package : mlpro.oa.tasks.driftdetectors.clusterbased.generic
## -- Module  : single_movement.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2025-03-04  0.1.0     DA/DS    Creation
## -- 2025-03-18  0.2.0     DA/DS    Completion of method DriftDetectorCBGenSingleMovement._get_drift_status()
## -------------------------------------------------------------------------------------------------

"""
Ver. 0.2.0 (2025-03-18)

This module provides a generic cluster-based drift detector for movement drift detection.
"""


from mlpro.bf.various import Log
from mlpro.bf.math.properties import *
from mlpro.oa.streams import OAStreamTask
from mlpro.oa.streams.tasks.clusteranalyzers import ClusterAnalyzer, Cluster
from mlpro.oa.streams.tasks.driftdetectors.clusterbased.generic.basics import DriftDetectorCBGenSingle
from mlpro.oa.streams.tasks.driftdetectors.drifts.clusterbased import DriftCBMovement



## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class DriftDetectorCBGenSingleMovement ( DriftDetectorCBGenSingle ):
    """
    Generic cluster-based drift detector for movement drift detection.

    Parameters
    ----------
    ...
    p_property : PropertyDefinition
        Cluster property to be observed.
    p_cls_drift : type
        Type of drift events to be raised.
    ...
    """

## -------------------------------------------------------------------------------------------------
    def __init__( self,
                  p_clusterer : ClusterAnalyzer,
                  p_property : PropertyDefinition,
                  p_thrs_lower : float,
                  p_thrs_upper : float,
                  p_cls_drift : type = DriftCBMovement,
                  p_name : str = None,
                  p_range_max = OAStreamTask.C_RANGE_THREAD,
                  p_ada : bool = True,
                  p_duplicate_data : bool = False,
                  p_visualize : bool = False,
                  p_logging=Log.C_LOG_ALL ):
        
        super().__init__( p_clusterer = p_clusterer,
                          p_property = p_property,
                          p_thrs_lower = p_thrs_lower,
                          p_thrs_upper = p_thrs_upper,
                          p_cls_drift = p_cls_drift,
                          p_name = p_name,
                          p_range_max = p_range_max,
                          p_ada = p_ada,
                          p_duplicate_data = p_duplicate_data,
                          p_visualize = p_visualize,
                          p_logging = p_logging )


## -------------------------------------------------------------------------------------------------
    def _get_drift_status( self, 
                           p_cluster : Cluster, 
                           p_properties : PropertyDefinitions, 
                           p_thrs_lower : float, 
                           p_thrs_upper = float, 
                           **p_kwargs ):
        
        # 1 Get property of interest from the cluster
        prop : Property = getattr( p_cluster, p_properties[0][0] )


        # 2 Get current drift status
        try:
            cluster_drifting = self.cluster_drifts[p_cluster.id].drift_status
        except:
            cluster_drifting = False


        # 3 Determine movement per dimension
        drift_status = False

        for d in range( prop.dim ):

            if ( cluster_drifting and ( prop.derivatives[1][d] > p_thrs_lower ) ) or \
               ( ( not cluster_drifting ) and ( prop.derivatives[1][d] > p_thrs_upper ) ):
            
                # 3.1 Cluster is drifting in this dimension
                drift_status = True
                break

        return drift_status