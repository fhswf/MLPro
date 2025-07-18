## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - The integrative middleware framework for standardized machine learning
## -- Package : mlpro.oa.tasks.changedetectors.anomalydetectors
## -- Module  : basics.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2023-06-08  0.0.0     SK       Creation
## -- 2023-09-12  1.0.0     SK       Release
## -- 2023-11-21  1.0.1     SK       Time Stamp update
## -- 2024-02-25  1.1.0     SK       Visualisation update
## -- 2024-04-10  1.2.0     DA/SK    Refactoring
## -- 2024-04-11  1.3.0     DA       Methods AnomalyDetector.init/update_plot: determination and
## --                                forwarding of changes on ax limits
## -- 2024-05-22  1.4.0     SK       Refactoring
## -- 2024-08-12  1.4.1     DA       Correction in AnomalyDetector.update_plot()
## -- 2024-12-11  1.4.2     DA       Pseudo classes if matplotlib is not installed
## -- 2025-02-14  1.5.0     DA       Review and refactoring
## -- 2025-03-03  1.5.1     DA       Corrections
## -- 2025-05-30  2.0.0     DA/DS    New parent class ChangeDetector
## -- 2025-06-06  2.1.0     DA       Refactoring: p_inst -> p_instances
## -- 2025-06-09  2.1.1     DA       Corrections in AnomalyDetector._triage()
## -- 2025-07-18  2.2.0     DA       Refactoring
## -------------------------------------------------------------------------------------------------

"""
Ver. 2.2.0 (2025-07-18)

This module provides templates for anomaly detection to be used in the context of online adaptivity.
"""


from mlpro.bf import Log
from mlpro.bf.streams import Instance

from mlpro.oa.streams import OAStreamTask
from mlpro.oa.streams.tasks.changedetectors import ChangeDetector
from mlpro.oa.streams.tasks.changedetectors.anomalydetectors.anomalies import Anomaly



# Export list for public API
__all__ = [ 'AnomalyDetector' ]




## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class AnomalyDetector (ChangeDetector):
    """
    Base class for online anomaly detectors. It raises an event when an anomaly is detected.

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
    p_kwargs : dict
        Further optional named parameters.
    """

    C_TYPE = 'Anomaly Detector'

## -------------------------------------------------------------------------------------------------
    def __init__( self,
                  p_name:str = None,
                  p_range_max = OAStreamTask.C_RANGE_THREAD,
                  p_ada : bool = True,
                  p_duplicate_data : bool = False,
                  p_visualize : bool = False,
                  p_logging=Log.C_LOG_ALL,
                  p_anomaly_buffer_size : int = 100,
                  p_thrs_inst : int = 0,
                  **p_kwargs ):

        super().__init__( p_name = p_name,
                          p_range_max = p_range_max,
                          p_ada = p_ada,
                          p_duplicate_data = p_duplicate_data,
                          p_visualize = p_visualize,
                          p_logging = p_logging,
                          p_change_buffer_size = p_anomaly_buffer_size,
                          p_thrs_inst = p_thrs_inst,
                          **p_kwargs )
        
        self.anomalies = self.changes


## -------------------------------------------------------------------------------------------------
    def _get_next_anomaly_id(self):
        """
        Methd that returns the id of the next anomaly. 

        Returns
        -------
        _ano_id : int
        """

        return self._get_next_change_id()


## -------------------------------------------------------------------------------------------------
    def _buffer_anomaly(self, p_anomaly:Anomaly):
        """
        Method to be used to add a new anomaly. Please use as part of your algorithm.

        Parameters
        ----------
        p_anomaly : Anomaly
            Anomaly object to be added.
        """

        self._buffer_change( p_change = p_anomaly )


## -------------------------------------------------------------------------------------------------
    def _remove_anomaly(self, p_anomaly:Anomaly):
        """
        Method to remove an existing anomaly. Please use as part of your algorithm.

        Parameters
        ----------
        p_anomaly : Anomaly
            Anomaly object to be removed.
        """

        self._remove_change( p_change = p_anomaly )


## -------------------------------------------------------------------------------------------------
    def _raise_anomaly_event( self, 
                              p_anomaly : Anomaly, 
                              p_instance : Instance, 
                              p_buffer: bool = True ):
        """
        Method to raise an anomaly event.

        Parameters
        ----------
        p_anomaly : Anomaly
            Anomaly object to be raised.
        p_instance : Instance = None
            Instance causing the anomaly. If provided, the time stamp of the instance is taken over
            to the anomaly.
        p_buffer : bool
            Anomaly is buffered when set to True.
        """

        self._raise_change_event( p_change = p_anomaly, 
                                  p_instance = p_instance,
                                  p_buffer = p_buffer )


## -------------------------------------------------------------------------------------------------
    def _triage(self, p_change, **p_kwargs) -> bool:
        return self._triage_anomaly (p_anomaly = p_change, **p_kwargs)


## -------------------------------------------------------------------------------------------------
    def _triage_anomaly( self, 
                         p_anomaly : Anomaly,
                         **p_kwargs ) -> bool:
        """
        Custom method for extended anomaly triage. Decides whether an already existing anomaly is 
        kept or removed. This method is called by the _run() method als part of its cleanup mechanism.

        Parameters
        ----------
        p_anomaly : Anomaly
            Anomaly object to be kept or discarded.
        **p_kwargs
            Optional keyword arguments (originally provided to the constructor).

        Returns
        -------
        bool
            True, if the specified anomaly shall be removed. False otherwise.
        """

        return False