## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - The integrative middleware framework for standardized machine learning
## -- Package : mlpro.bf
## -- Module  : ops
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2022-10-28  0.0.0     DA       Creation 
## -- 2022-10-29  1.0.0     DA       Implementation of classes Mode, ScenarioBase 
## -- 2022-10-31  1.1.0     DA       Class ScenarioBase: plot functionality added 
## -- 2022-11-07  1.2.0     DA       Class ScenarioBase: 
## --                                - support of new event "end of data"
## --                                - method setup(): parameters removed
## -- 2022-11-12  1.2.1     DA       Class ScenarioBase: minor changes on logging
## -- 2022-11-21  1.2.2     DA       Eliminated all uses of super()
## -- 2023-03-25  1.2.3     DA       Class ScenarioBase: new parent class Persistent
## -- 2024-11-09  1.3.0     DA       Class ScenarioBase: new parent class KWArgs
## -- 2024-12-29  1.4.0     DA       Method ScenarioBase.run(): logging of duration/speed
## -- 2025-07-18  1.5.0     DA       Refactoring
## -------------------------------------------------------------------------------------------------

"""
Ver. 1.5.0 (2025-07-18)

This module provides classes for operation.
"""


import sys
from datetime import timedelta, datetime
from mlpro.bf.various import Log, Persistent, Timer, KWArgs
from mlpro.bf.plot import Plottable
from mlpro.bf.events import *
from mlpro.bf.exceptions import ParamError



# Export list for public API
__all__ = [ 'Mode',
            'ScenarioBase' ]




## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class Mode (Log):
    """
    Property class that adds a mode and related methods to a child class.

    Parameters
    ----------
    p_mode
        Operation mode. Valid values are stored in constant C_VALID_MODES.
    p_logging
        Log level (see constants of class Log). Default: Log.C_LOG_ALL

    Attributes
    ----------
    C_MODE_SIM = 0
        Simulation mode.
    C_MODE_REAL = 1
        Real operation mode.
    C_VALID_MODES : list
        List of valid modes.
    """

    C_MODE_INITIAL  = -1
    C_MODE_SIM      = 0
    C_MODE_REAL     = 1

    C_VALID_MODES   = [ C_MODE_SIM, C_MODE_REAL ]

## -------------------------------------------------------------------------------------------------
    def __init__(self, p_mode, p_logging=Log.C_LOG_ALL):
        Log.__init__(self, p_logging=p_logging)
        if not p_mode in self.C_VALID_MODES: raise ParamError('Invalid mode')
        self._mode = p_mode


## -------------------------------------------------------------------------------------------------
    def get_mode(self):
        """
        Returns current mode.
        """

        return self._mode


## -------------------------------------------------------------------------------------------------
    def set_mode(self, p_mode):
        """
        Sets new mode.

        Parameters
        ----------
        p_mode
            Operation mode. Valid values are stored in constant C_VALID_MODES.
        """

        if not p_mode in self.C_VALID_MODES: raise ParamError('Invalid mode')
        if self._mode == p_mode: return
        self._mode = p_mode
        self.log(self.C_LOG_TYPE_I, 'Operation mode set to', self._mode)





## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class ScenarioBase (Mode, Persistent, Plottable, KWArgs):
    """
    Base class for executable scenarios in MLPro. To be inherited and specialized in higher layers.
    
    The following key features are included:
      - Operation mode
      - Cycle management
      - Timer
      - Latency 

    Parameters
    ----------
    p_mode
        Operation mode. See Mode.C_VALID_MODES for valid values. Default = Mode.C_MODE_SIM.
    p_id
        Optional external id
    p_cycle_limit : int
        Maximum number of cycles. Default = 0 (no limit).
    p_auto_setup : bool
        If True custom method setup() is called after initialization.
    p_visualize : bool
        Boolean switch for visualisation. Default = True.
    p_logging
        Log level (see constants of class Log). Default: Log.C_LOG_ALL.  
    p_kwargs : dict
        Custom keyword parameters handed over to custom method setup().
    """

    C_TYPE          = 'Scenario Base'
    C_NAME          = '????'

## -------------------------------------------------------------------------------------------------
    def __init__(self, 
                 p_mode,       
                 p_id = None,
                 p_cycle_limit=0,  
                 p_auto_setup:bool = True,
                 p_visualize:bool=True,              
                 p_logging=Log.C_LOG_ALL,
                 **p_kwargs ):    

        # 1 Initialization
        KWArgs.__init__( self, **p_kwargs )
        Persistent.__init__( self, p_id = p_id, p_logging = p_logging )
        Mode.__init__( self, p_mode, p_logging )
        Plottable.__init__( self, p_visualize = p_visualize )
        self._cycle_max     = sys.maxsize
        self._cycle_id      = 0
        self._visualize     = p_visualize
        self._cycle_limit   = p_cycle_limit
        self._timer         = None

        # 2 Optional automatic custom setup
        if p_auto_setup: self.setup( **self._get_kwargs() )
        

## -------------------------------------------------------------------------------------------------
    def _init_timer(self):
        if self.get_mode() == Mode.C_MODE_SIM:
            t_mode = Timer.C_MODE_VIRTUAL
        else:
            t_mode = Timer.C_MODE_REAL

        self._timer     = Timer(t_mode, self.get_latency(), self._cycle_limit)


## -------------------------------------------------------------------------------------------------
    def setup(self, **p_kwargs):
        """
        Custom method to set up all components of the scenario.

        Parameters
        ----------
        p_kwargs : dict
            Custom keyword parameters.
        """

        raise NotImplementedError


## -------------------------------------------------------------------------------------------------
    def set_mode(self, p_mode):
        """
        Sets operation mode of the scenario. Custom method _set_mode() is called.

        Parameter
        ---------
        p_mode
            Operation mode. See class bf.ops.Mode for further details.
        """

        Mode.set_mode(self, p_mode=p_mode)
        self._set_mode(p_mode)
        self._timer = None


## -------------------------------------------------------------------------------------------------
    def _set_mode(self, p_mode):
        """
        Custom method to set the operation mode of components of the scenario. See method set_mode()
        for further details.

        Parameter
        ---------
        p_mode
            Operation mode. See class bf.ops.Mode for further details.
        """

        raise NotImplementedError


## -------------------------------------------------------------------------------------------------
    def get_latency(self) -> timedelta:
        """
        Returns the latency of the scenario. To be implemented in child class.
        """

        raise NotImplementedError


## -------------------------------------------------------------------------------------------------
    def set_cycle_limit(self, p_limit):
        """
        Sets the maximum number of cycles to run.

        Parameters
        ----------
        p_cycle_limit : int
            Maximum number of cycles. Default = 0 (no limit).
        """

        self._cycle_limit = p_limit


## -------------------------------------------------------------------------------------------------
    def reset(self, p_seed=1):
        """
        Resets the scenario and especially the ML model inside. Internal random generators are seed 
        with the given value. Custom reset actions can be implemented in method _reset().

        Parameters
        ----------
        p_seed : int          
            Seed value for internal random generator
        """

        # 1 Intro
        if self._timer is None: self._init_timer()
        self.log(self.C_LOG_TYPE_I, 'Process time', self._timer.get_time(), ': Scenario reset with seed', str(p_seed))

        # 2 Custom reset of further scenario-specific components
        self._reset(p_seed)

        # 3 Timer reset
        self._timer.reset()

        # 4 Cycle counter reset
        self._cycle_id = 0


## -------------------------------------------------------------------------------------------------
    def _reset(self, p_seed):
        """
        Custom method to reset the components of the scenario and to set the given random seed value. 
        See method reset() for further details.

        Parameters
        ----------
        p_seed : int          
            Seed value for internal random generator
        """

        pass


## -------------------------------------------------------------------------------------------------
    def run_cycle(self):
        """
        Runs a single process cycle.

        Returns
        -------
        success : bool
            True on success. False otherwise.
        error : bool
            True on error. False otherwise.
        timeout : bool
            True on timeout. False otherwise.
        cycle_limit : bool
            True, if cycle limit has reached. False otherwise.
        adapted : bool
            True, if something within the scenario has adapted something in this cycle. False otherwise.
        end_of_data : bool
            True, if the end of the related data source has been reached. False otherwise.
        """

        # 0 Initialization
        limit = timeout = False
        if self._timer is None: self._init_timer()


        # 1 Run a single custom cycle
        self.log(self.C_LOG_TYPE_S, 'Process time', self._timer.get_time(), ': Start of cycle', str(self._cycle_id))
        success, error, adapted, end_of_data = self._run_cycle()
        self.log(self.C_LOG_TYPE_S, 'Process time', self._timer.get_time(), ': End of cycle', str(self._cycle_id))


        # 2 End of data source reached?
        if end_of_data:
            self.log(self.C_LOG_TYPE_S, 'Process time', self._timer.get_time(), ': End of data source reached')
            return success, error, timeout, limit, adapted, end_of_data


        # 3 Update visualization

        if self._visualize:
            self.update_plot()


        # 4 Update cycle id and check for optional limit
        if ( self._cycle_limit > 0 ) and ( self._cycle_id >= ( self._cycle_limit -1 ) ): 
            limit = True
        else:
            self._cycle_id = ( self._cycle_id + 1 ) & self._cycle_max


        # 5 Wait for next cycle (real mode only)
        if ( self._timer.finish_lap() == False ) and ( self._cycle_len is not None ):
            self.log(self.C_LOG_TYPE_W, 'Process time', self._timer.get_time(), ': Process timed out !!!')
            timeout = True


        # 6 Return result of custom cycle execution
        return success, error, timeout, limit, adapted, end_of_data


## -------------------------------------------------------------------------------------------------
    def _run_cycle(self):
        """
        Custom implementation of a single process cycle. To be redefined.

        Returns
        -------
        success : bool
            True on success. False otherwise.
        error : bool
            True on error. False otherwise.
        adapted : bool
            True, if something within the scenario has adapted something in this cycle. False otherwise.
        end_of_data : bool
            True, if the end of the related data source has been reached. False otherwise.
        """

        raise NotImplementedError


## -------------------------------------------------------------------------------------------------
    def get_cycle_id(self):
        """
        Returns current cycle id.
        """

        return self._cycle_id


## -------------------------------------------------------------------------------------------------
    def run(self, 
            p_term_on_success:bool=True,        
            p_term_on_error:bool=True,          
            p_term_on_timeout:bool=False ):    
        """
        Runs the scenario as a sequence of single process steps until a terminating event occures.

        Parameters
        ----------
        p_term_on_success : bool
            If True, the run terminates on success. Default = True.
        p_term_on_error : bool
            If True, the run terminates on error. Default = True.
        p_term_on_timeout : bool
            If True, the run terminates on timeout. Default = False.

        Returns
        -------
        success : bool
            True on success. False otherwise.
        error : bool
            True on error. False otherwise.
        timeout : bool
            True on timeout. False otherwise.
        cycle_limit : bool
            True, if cycle limit has reached. False otherwise.
        adapted : bool
            True, if ml model adapted something. False otherwise.
        end_of_data : bool
            True, if the end of the related data source has been reached. False otherwise.
        num_cycles: int
            Number of cycles.
        """

        #  1 Intro
        self._cycle_id  = 0
        adapted         = False
        if self._timer is None: self._init_timer()
        self.log(self.C_LOG_TYPE_S, 'Process time', self._timer.get_time(), ': Start of processing')


        # 2 Late initialization of visualization with default parameters
        if self.get_visualization():
            self.init_plot()


        # 3 Time measurement, if logging is active and a cycle limit is set
        if ( self.get_log_level() in [ Log.C_LOG_ALL, Log.C_LOG_WE ] ) and ( self._cycle_limit > 0 ):
            tp_before = datetime.now()


        # 4 Main loop 
        while True:
            success, error, timeout, limit, adapted_cycle, end_of_data = self.run_cycle()
            adapted = adapted or adapted_cycle
            if p_term_on_success and success: break
            if p_term_on_error and error: break
            if p_term_on_timeout and timeout: break
            if limit or end_of_data: break


        # 5 Outro
        self.log(self.C_LOG_TYPE_S, 'Process time', self._timer.get_time(), ': End of processing')

        if ( self.get_log_level() in [ Log.C_LOG_ALL, Log.C_LOG_WE ] ) and ( self._cycle_limit > 0 ):
            tp_after = datetime.now()
            tp_delta = tp_after - tp_before
            duration_musec = ( tp_delta.seconds * 1000000 + tp_delta.microseconds )
            if duration_musec == 0: duration_musec = 1
            duration_sec = duration_musec / 1000000

            self.log(Log.C_LOG_TYPE_W, str(self._cycle_id + 1),'cycles in', round(duration_sec,2), 's (' + str(round( (self._cycle_id + 1) / duration_sec,1)), 'cycles/s)', p_type_col='S')

        return success, error, timeout, limit, adapted, end_of_data, self._cycle_id