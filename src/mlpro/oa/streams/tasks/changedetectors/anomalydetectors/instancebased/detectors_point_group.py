## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - The integrative middleware framework for standardized machine learning
## -- Package : mlpro.oa.streams.tasks.anomalydetectors.instancebased
## -- Module  : detectors_point_group.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2023-06-08  0.0.0     SK       Creation
## -- 2023-09-12  1.0.0     SK       Release
## -- 2023-11-21  1.0.1     SK       Time Stamp update
## -- 2024-02-25  1.1.0     SK       Visualisation update
## -- 2024-04-10  1.2.0     DA/SK    Code review
## -- 2024-05-07  1.2.1     SK       Bug fix on groupanomaly visualisation
## -- 2024-08-12  1.3.0     DA       Review and adjustments on documentation
## -- 2025-02-14  1.4.0     DA       Refactoring
## -- 2025-02-17  1.5.0     DA       Review and generalization
## -- 2025-03-05  1.6.0     DA       Refactoring and simplification
## -- 2025-06-08  1.6.1     DA       Review/refactoring
## -- 2025-06-13  1.6.2     DA       Bugfix in method _buffer_anomaly()
## -- 2025-07-18  1.7.0     DA       Refactoring
## -------------------------------------------------------------------------------------------------

"""
Ver. 1.7.0 (2025-07-18)

This module provides an extended template for instance-based anomaly detectors that supports an optional
group anomaly detection based on point anomalies.
"""


from mlpro.bf import Log
from mlpro.bf.streams import Instance

from mlpro.oa.streams.basics import StreamTask
from mlpro.oa.streams.tasks.changedetectors.anomalydetectors.instancebased.basics import AnomalyDetectorIB
from mlpro.oa.streams.tasks.changedetectors.anomalydetectors.anomalies import Anomaly, PointAnomaly, GroupAnomaly



# Export list for public API
__all__ = [ 'AnomalyDetectorIBPG' ]





## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class AnomalyDetectorIBPG (AnomalyDetectorIB):
    """
    This class is an extended template offering an optional group anomaly detection based on point 
    anomalies. This detection can be turned on/off via the p_group_anomaly_det parameter. See 
    method _buffer_anomaly() for further details. 

    Parameters
    ----------
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
    p_anomaly_buffer_size : int = 100
        Size of the internal anomaly buffer self.anomalies. Default = 100.
    p_thrs_inst : int = 0
        The algorithm is only executed after this number of instances.
    p_group_anomaly_det : bool = True
        Paramter to activate group anomaly detection. Default is True.
    p_kwargs : dict
        Further optional named parameters.
    """

    C_TYPE = 'Anomaly Detector (IBPG)'

## -------------------------------------------------------------------------------------------------
    def __init__( self,
                  p_name:str = None,
                  p_range_max = StreamTask.C_RANGE_THREAD,
                  p_ada : bool = True,
                  p_duplicate_data : bool = False,
                  p_visualize : bool = False,
                  p_logging = Log.C_LOG_ALL,
                  p_anomaly_buffer_size : int = 100,
                  p_thrs_inst : int = 0,
                  p_group_anomaly_det : bool = True,
                  **p_kwargs ):

        super().__init__( p_name = p_name,
                          p_range_max = p_range_max,
                          p_ada = p_ada,
                          p_duplicate_data = p_duplicate_data,
                          p_visualize = p_visualize,
                          p_logging = p_logging,
                          p_anomaly_buffer_size = p_anomaly_buffer_size,
                          p_thrs_inst = p_thrs_inst,
                          **p_kwargs )
        
        self._group_anomalies : list[Anomaly] = []
        self._group_anomalies_instances : list[Instance] = []
        self._group_anomaly_det = p_group_anomaly_det


## -------------------------------------------------------------------------------------------------
    def _buffer_anomaly(self, p_anomaly : Anomaly):
        """
        Method to be used to add a new anomaly. Please use as part of your algorithm. 

        Parameters
        ----------
        p_anomaly : Anomaly
            Anomaly object to be added.
        """

        if ( not self._group_anomaly_det ) or ( type(p_anomaly) != PointAnomaly ):
            return super()._buffer_anomaly( p_anomaly = p_anomaly)


        if self._group_anomaly_det:
            self._group_anomalies.append(p_anomaly)
            self._group_anomalies_instances.append(p_anomaly.instances[-1])

            if len(self._group_anomalies_instances) > 1:

                inst_2 = self._group_anomalies_instances[-1]
                second = inst_2.id
                inst_1 = self._group_anomalies_instances[-2]
                first  = inst_1.id
                    
                if int(second) - 1 == int(first):

                    if len(self._group_anomalies_instances) == 3:

                        for i in range(2):
                            self._remove_anomaly(self._group_anomalies[i])

                        self._ano_id -= 2
                        groupanomaly = GroupAnomaly( p_instances = self._group_anomalies_instances,
                                                     p_visualize=self.get_visualization(),
                                                     p_raising_object = self,
                                                     p_tstamp = inst_2.tstamp )
                        
                        self._raise_anomaly_event( p_anomaly = groupanomaly )
                            
                        # super()._buffer_anomaly( p_anomaly = groupanomaly )                    

                        self._group_anomalies = []
                        self._group_anomalies.append(groupanomaly)

                    elif len(self._group_anomalies_instances) > 3:
                        self._group_anomalies[0].instances = self._group_anomalies_instances
                        self._group_anomalies.pop(-1)
                        return self._group_anomalies[0]
                            
                    else:
                        super()._buffer_anomaly( p_anomaly = p_anomaly)                    

                else:
                    for groupanomaly in self._group_anomalies:
                        if isinstance(groupanomaly, GroupAnomaly):
                            groupanomaly.plot_update = False
                    self._group_anomalies = []
                    self._group_anomalies_instances = []
                    self._group_anomalies.append(p_anomaly)
                    self._group_anomalies_instances.append(p_anomaly.instances[-1])
                    super()._buffer_anomaly( p_anomaly = p_anomaly )

            else:
                super()._buffer_anomaly( p_anomaly = p_anomaly)            