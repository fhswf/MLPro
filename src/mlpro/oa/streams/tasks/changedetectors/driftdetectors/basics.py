## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - The integrative middleware framework for standardized machine learning
## -- Package : mlpro.oa.streams.tasks.changedetectors.driftdetectors
## -- Module  : basics.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2025-02-12  0.1.0     DA       Creation
## -- 2025-03-03  0.2.0     DA       Alignment with anomaly detection
## -- 2025-05-30  1.0.0     DA/DS    Class DriftDetector: new parent ChangeDetector
## -- 2025-06-06  1.1.0     DA       Refactoring: p_inst -> p_instances
## -- 2025-06-09  1.1.1     DA       Corrections in DriftDetector._triage()
## -- 2025-07-18  1.2.0     DA       Refactoring
## -------------------------------------------------------------------------------------------------

"""
Ver. 1.2.0 (2025-07-18)

This module provides templates for drift detection to be used in the context of online adaptivity.
"""


from mlpro.bf import Log
from mlpro.bf.streams import Instance
from mlpro.oa.streams import OAStreamTask
from mlpro.oa.streams.tasks.changedetectors.driftdetectors.drifts import Drift
from mlpro.oa.streams.tasks.changedetectors import ChangeDetector



# Export list for public API
__all__ = [ 'DriftDetector' ]




## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class DriftDetector (ChangeDetector):
    """
    Base class for online anomaly detectors. It raises an event whenever the beginning or the end 
    of a drift is detected. Please describe in child classes which event classes are used. Always
    use the _raise_drift_event() method when raising an event. 

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
    p_drift_buffer_size : int = 100
        Size of the internal drift buffer self.drifts. Default = 100.
    p_thrs_inst : int = 0
        The algorithm is only executed after this number of instances.
    p_kwargs : dict
        Further optional named parameters.
    """

    C_TYPE = 'Drift Detector'

## -------------------------------------------------------------------------------------------------
    def __init__( self,
                  p_name:str = None,
                  p_range_max = OAStreamTask.C_RANGE_THREAD,
                  p_ada : bool = True,
                  p_duplicate_data : bool = False,
                  p_visualize : bool = False,
                  p_logging=Log.C_LOG_ALL,
                  p_drift_buffer_size : int = 100,
                  p_thrs_inst : int = 0,
                  **p_kwargs ):

        super().__init__( p_name = p_name,
                          p_range_max = p_range_max,
                          p_ada = p_ada,
                          p_duplicate_data = p_duplicate_data,
                          p_visualize = p_visualize,
                          p_logging = p_logging,
                          p_change_buffer_size = p_drift_buffer_size,
                          p_thrs_inst = p_thrs_inst,
                          **p_kwargs )
        
        self.drifts = self.changes
        

## -------------------------------------------------------------------------------------------------
    def _get_next_drift_id(self):
        """
        Methd that returns the id of the next drift. 

        Returns
        -------
        drift_id : int
        """

        return self._get_next_change_id()


## -------------------------------------------------------------------------------------------------
    def _buffer_drift(self, p_drift : Drift):
        """
        Method to be used internally to add a new drift object. Please use as part of your algorithm.

        Parameters
        ----------
        p_drift : Drift
            Drift object to be added.
        """

        self._buffer_change( p_change = p_drift )


## -------------------------------------------------------------------------------------------------
    def _remove_drift(self, p_drift : Drift):
        """
        Method to remove an existing drift object. Please use as part of your algorithm.

        Parameters
        ----------
        p_drift : Drift
            Drift object to be removed.
        """

        self._remove_change( p_change = p_drift )


## -------------------------------------------------------------------------------------------------
    def _raise_drift_event( self, 
                            p_drift : Drift, 
                            p_instance : Instance = None,
                            p_buffer: bool = True ):
        """
        Specialized method to raise drift events. 

        Parameters
        ----------
        p_drift : Drift
            Drift event object to be raised.
        p_instance : Instance = None
            Instance causing the drift. If provided, the time stamp of the instance is taken over
            to the drift.
        p_buffer : bool
            Drift is buffered when set to True.
        """

        self._raise_change_event( p_change = p_drift,
                                  p_instance = p_instance,
                                  p_buffer = p_buffer )
        

## -------------------------------------------------------------------------------------------------
    def _triage(self, p_change, **p_kwargs) -> bool:
        return self._triage_drift (p_drift = p_change, **p_kwargs)


## -------------------------------------------------------------------------------------------------
    def _triage_drift( self, 
                       p_drift : Drift,
                       **p_kwargs ) -> bool:
        """
        Custom method for extended drift triage. Decides whether an already existing drift is kept or removed.
        This method is called by the _run() method als part of its cleanup mechanism.

        Parameters
        ----------
        p_drift : Drift
            Drift object to be kept or discarded.
        **p_kwargs
            Optional keyword arguments (originally provided to the constructor).

        Returns
        -------
        bool
            True, if the specified drift shall be removed. False otherwise.
        """

        return False