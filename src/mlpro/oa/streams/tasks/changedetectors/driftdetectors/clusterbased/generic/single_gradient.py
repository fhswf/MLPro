## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - The integrative middleware framework for standardized machine learning
## -- Package : mlpro.oa.tasks.driftdetectors.clusterbased.generic
## -- Module  : single_movement.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2025-03-04  0.1.0     DA/DS    Creation
## -- 2025-03-18  0.2.0     DA/DS    Completion of method _get_drift_status()
## -- 2025-03-26  0.3.0     DA       Method _get_drift_status(): exception if property is misdefined
## -- 2025-05-06  0.3.1     DA       Bugfix in method _get_drift-status()
## -- 2025-05-20  0.3.2     DA/DS    Bugfixes
## -- 2025-06-10  0.4.0     DA/DS    New class name: DriftDetectorCBGenSingleGradient
## -- 2025-06-11  0.4.1     DA       Bugfixes 
## -- 2025-07-18  0.5.0     DA       Refactoring
## -- 2025-09-30  0.5.1     DA/DS    Bugfix in DriftDetectorCBGenSingleGradient._get_drift_status()
## -- 2025-10-02  0.5.2     DA/DS    Bugfix in DriftDetectorCBGenSingleGradient._get_drift_status()
## -------------------------------------------------------------------------------------------------

"""
Ver. 0.5.2 (2025-10-02) 

This module provides a generic cluster-based drift detector for movement drift detection.
"""

import numpy as np

from mlpro.bf import Log, ImplementationError
from mlpro.bf.math.properties import *
from mlpro.oa.streams import OAStreamTask
from mlpro.oa.streams.tasks.clusteranalyzers import ClusterAnalyzer, Cluster
from mlpro.oa.streams.tasks.changedetectors.driftdetectors.clusterbased.generic.basics import DriftDetectorCBGeneric
from mlpro.oa.streams.tasks.changedetectors.driftdetectors.drifts.clusterbased import DriftCBMovement



# Export list for public API
__all__ = [ 'DriftDetectorCBGenSingleGradient' ]




## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class DriftDetectorCBGenSingleGradient ( DriftDetectorCBGeneric ):
    """
    Generic cluster-based drift detector for gradient changes of single properties.

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
                  p_drift_buffer_size : int = 100,
                  p_thrs_inst : int = 0,
                  p_thrs_clusters : int = 1,
                  p_logging=Log.C_LOG_ALL ):
        
        super().__init__( p_clusterer = p_clusterer,
                          p_properties = [p_property] ,
                          p_thrs_lower = p_thrs_lower,
                          p_thrs_upper = p_thrs_upper,
                          p_cls_drift = p_cls_drift,
                          p_name = p_name,
                          p_range_max = p_range_max,
                          p_ada = p_ada,
                          p_duplicate_data = p_duplicate_data,
                          p_visualize = p_visualize,
                          p_drift_buffer_size = p_drift_buffer_size,
                          p_thrs_inst = p_thrs_inst,
                          p_thrs_clusters = p_thrs_clusters,
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


        # 2 Get the absolute first order derivative from the property
        try:
            derivative_o1 = prop.derivatives[1]
            if derivative_o1 is not None:
                abs_derivative_o1 = abs( derivative_o1 )
            else:
                return False

        except:
            if prop._derivative_order_max == 0:
                raise ImplementationError('MLPro: Cluster property "' + p_properties[0][0] + '" needs to provide a maximum derivative order > 0')

            return False

        if prop.dim == 1:
            abs_derivative_o1 = [ abs_derivative_o1 ]
        

        # 3 Get current drift status
        

        # 4 Determine movement per dimension
        status = False

        for d in range( prop.dim ):

            if ( p_thrs_lower < (abs_derivative_o1[d])  and ( ( p_thrs_upper == 0 ) or (abs_derivative_o1[d]) < p_thrs_upper ) ):
            
                # 4.1 Cluster is drifting in this dimension
                status = True
                break

        return status