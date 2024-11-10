## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - The integrative middleware framework for standardized machine learning
## -- Package : mlpro.bf.control.control_scenarios
## -- Module  : basic.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2024-10-04  0.1.0     DA       Initial implementation 
## -- 2024-10-09  0.2.0     DA       Refactoring
<<<<<<< HEAD
## -- 2024-11-04  0.3.0     ASP       Implementation class CascadeControlSystem
=======
## -- 2024-11-09  0.3.0     DA       Refactoring
>>>>>>> origin/bf/oa/control
## -------------------------------------------------------------------------------------------------

"""
Ver. 0.3.0 (2024-11-09)

This module provides a simplified container class for a basic synchronous control system containing

- a controller
- a controlled system
- an optional integrator for the control variable

"""

from typing import Union

from mlpro.bf.various import Log
<<<<<<< HEAD
from mlpro.bf.control import Controller, ControlledSystem, ControlSystem,CascadedSystem
from mlpro.bf.control.operators import Comparator, Integrator,Converter
from mlpro.bf.control.basics import ControlledVariable,ControlVariable,SetPoint
from mlpro.bf.math import *
=======
from mlpro.bf.systems import System
from mlpro.bf.control import Controller, ControlledSystem
from mlpro.bf.control.controlsystems import CascadeControlSystem
from mlpro.bf.control.operators import Integrator

>>>>>>> origin/bf/oa/control



## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
<<<<<<< HEAD
class ControlSystemBasic (ControlSystem):
    
=======
class BasicControlSystem (CascadeControlSystem):
>>>>>>> origin/bf/oa/control
    """
    Simplified container class for a basic synchronous control system containing

    - a controller
    - a controlled system
    - an optional integrator for the control variable

    Parameters
    ----------
    p_mode
        Operation mode. See Mode.C_VALID_MODES for valid values. Default = Mode.C_MODE_SIM.
    p_controller : Controller
        Controller to be used in the control workflow
    p_controlled_system : ControlledSystem
        Controlled system to be used in the control workflow
    p_name : str = ''
        Name of the control system
    p_cycle_limit : int
        Maximum number of cycles. Default = 0 (no limit).
    p_ctrl_var_integration : bool = False
        If True, an optional intrator is added to control workflow
    p_visualize : bool
        Boolean switch for visualisation. Default = False.
    p_logging
        Log level (see constants of class Log). Default: Log.C_LOG_ALL.  
    p_kwargs : dict
        Custom keyword parameters handed over to custom method setup().
    """

    C_TYPE          = 'Basic Control System'

## -------------------------------------------------------------------------------------------------
    def __init__( self, 
                  p_mode, 
                  p_controller : Controller,
                  p_controlled_system : Union[System, ControlledSystem],
                  p_ctrl_var_integration : bool = False,
                  p_name : str = '',
                  p_cycle_limit = 0, 
                  p_visualize : bool = False, 
                  p_logging = Log.C_LOG_ALL,
                  **p_kwargs ):
        
        if p_ctrl_var_integration:
            controllers= [ [ p_controller, 
                             Integrator( p_range_max = p_controller.get_range(),
                                         p_visualize = p_visualize,
                                         p_logging = p_logging ) ] ]
        else:
            controllers = [ p_controller ]

        
        super().__init__( p_mode = p_mode,
                          p_controllers = controllers,
                          p_controlled_systems = [ p_controlled_system ],
                          p_name = p_name,
                          p_cycle_limit = p_cycle_limit,
                          p_visualize = p_visualize,
<<<<<<< HEAD
                          p_logging = p_logging )
        

## -------------------------------------------------------------------------------------------------
    def _setup(self, p_mode, p_visualize: bool, p_logging) -> ControlWorkflow:
        
        # 1 Create a new control cycle
        control_workflow = ControlWorkflow( p_mode = p_mode,
                                            p_visualize = p_visualize,
                                            p_logging = p_logging )
        

        # 2 Create and add a comparator
        comparator = Comparator( p_visualize = p_visualize, p_logging = p_logging )
        control_workflow.add_task( p_task = comparator )


        # 3 Add the controller
        control_workflow.add_task( p_task = self._controller, p_pred_tasks = [comparator] )


        # 4 Optionally create and add an integrator
        if self._ctrl_var_integration:
            integrator = Integrator( p_visualize = p_visualize, p_logging = p_logging )
            control_workflow.add_task( p_task = integrator, p_pred_tasks = [self._controller] )
            pred_sys = integrator

        else:
            pred_sys = self._controller


        # 5 Add the controlled system
        control_workflow.add_task( p_task = self._controlled_system, p_pred_tasks = [pred_sys] )
        self._controlled_system.system.set_mode( p_mode = p_mode )


        # 6 Initialize and return the prepared control workflow
        control_workflow.get_so().init( p_ctrlled_var_space = self._controlled_system.system.get_state_space(),
                                        p_ctrl_var_space = self._controlled_system.system.get_action_space() )
        
        return control_workflow
    

## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class CascadeControlSystem (ControlSystem):
    
    """
    Simplified container class for a basic synchronous cascade control system containing

    - list of controller
    - list controlled system
    - an optional integrator for the control variable

    Parameters
    ----------
    p_controllers : list[Controller]
        Controllers to be used in the control workflow
    p_controlled_systems : list[ControlledSystem]
        Controlled systems to be used in the control workflow
    p_ctrl_var_integration : bool = False
        If True, an optional intrator is added to control workflow
    """

    C_TYPE          = 'Cascade Control System'

## -------------------------------------------------------------------------------------------------
    def __init__( self, 
                  p_controllers : list[Controller],                 
                  p_cascaded_system : list[CascadedSystem],
                  p_mode, 
                  p_ctrl_var_integration : bool = False,
                  p_cycle_limit = 0, 
                  p_visualize : bool = False, 
                  p_logging = Log.C_LOG_ALL ):

        self._control_loops = len(p_controllers)        
        self._controllers           = p_controllers
        self._cascaded_system    = p_cascaded_system
        self._ctrl_var_integration = p_ctrl_var_integration

        super().__init__( p_mode = p_mode,
                          p_cycle_limit = p_cycle_limit,
                          p_visualize = p_visualize,
                          p_logging = p_logging )
        

## -------------------------------------------------------------------------------------------------
    def _setup(self, p_mode, p_visualize: bool, p_logging) -> ControlWorkflow:
        
        # 1 Create a new control cycle
        control_workflow = ControlWorkflow( p_mode = p_mode,
                                            p_visualize = p_visualize,
                                            p_logging = p_logging )
        
        
        num_converters = self._control_loops-1
        pred_sys = None

        # 2 Add Controllers to the workflow
        for controller in self._controllers:            
            comparator = Comparator( p_visualize = p_visualize, p_logging = p_logging )
            
            if pred_sys == None:
                control_workflow.add_task( p_task = comparator )
            else:
                control_workflow.add_task( p_task = comparator, p_pred_tasks = [pred_sys] )
 
            control_workflow.add_task( p_task = controller, p_pred_tasks = [comparator] )
            pred_sys = controller

            #Add a converter Task, if more than one controller exist. ControlVariable --> SetPoint
            if num_converters >0:
                converter = Converter(p_src_type=ControlVariable,p_dst_type=SetPoint,p_visualize = p_visualize, p_logging = p_logging )
                control_workflow.add_task(p_task=converter,p_pred_tasks=[pred_sys])
                num_converters-=1
                pred_sys = converter
            
        num_converters = self._control_loops-1 
        m_state_space = MSpace()
        m_action_space= MSpace()


        # 3 Add the controlled systems to the workflow
        for idx in range(1,self._control_loops+1):
            self._cascaded_system[-idx].system.set_mode(p_mode = p_mode)
            m_state_space.append(p_set= self._cascaded_system[-idx].system.get_state_space())
            m_action_space.append(p_set= self._cascaded_system[-idx].system.get_action_space())
            control_workflow.add_task( p_task =self._cascaded_system[-idx], p_pred_tasks = [pred_sys] )
            pred_sys= self._cascaded_system[-idx]        

            
            #Add a converter Task, if more than one cotrolled system exist. ControlledVariable--> ControlVariable
            if num_converters >0:
                converter = Converter(p_src_type=ControlledVariable,p_dst_type=ControlVariable,p_visualize = p_visualize, p_logging = p_logging )
                control_workflow.add_task(p_task=converter,p_pred_tasks=[pred_sys])
                num_converters-=1
                pred_sys = converter  

        
        # 4 Initialize and return the prepared control workflow
        control_workflow.get_so().init( p_ctrlled_var_space = m_state_space,
                                        p_ctrl_var_space = m_action_space)
        
        
        return control_workflow
=======
                          p_logging = p_logging,
                          **p_kwargs )
>>>>>>> origin/bf/oa/control
