## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - The integrative middleware framework for standardized machine learning
## -- Package : mlpro.bf.control.controlsystems
## -- Module  : cascade.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2024-11-08  0.1.0     DA       Initial implementation
## -- 2024-11-09  1.0.0     DA       Extended ControlledSystemList by type System
## -- 2025-07-18  1.1.0     DA       Refactoring
## -------------------------------------------------------------------------------------------------

"""
Ver. 1.1.0 (2025-07-18)

This module provides a container class for cascade control systems.

"""

from typing import List, Union

from mlpro.bf import Log
from mlpro.bf.exceptions import *
from mlpro.bf.systems import System
from mlpro.bf.control.basics import *
from mlpro.bf.control.operators import Comparator, Converter



# Export list for public API
__all__ = [ 'ControllerList', 
            'ControlledSystemList', 
            'CascadeControlSystem' ]




ControllerList       = List[ Union[ Controller, List[ControlTask] ] ]
ControlledSystemList = List[ Union[ System, ControlledSystem, List[ControlTask] ] ]




## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class CascadeControlSystem (ControlSystem):
    """
    Cascade control system.

    Parameters
    ----------
    p_mode
        Operation mode. See Mode.C_VALID_MODES for valid values. Default = Mode.C_MODE_SIM.
    p_controllers : ControllerList
        List of controllers to be cascaded in order to outer to inner controller.
    p_controlled_systems : ControlledSystemLists
        List of controlled systems to be cascaded in order to outer to inner controlled system.
    p_name : str = ''
        Name of the control system
    p_cycle_limit : int
        Maximum number of cycles. Default = 0 (no limit).
    p_visualize : bool
        Boolean switch for visualisation. Default = False.
    p_logging
        Log level (see constants of class Log). Default: Log.C_LOG_ALL.  
    p_kwargs : dict
        Custom keyword parameters handed over to custom method setup().
    """

    C_TYPE          = 'Cascade Control System'

## -------------------------------------------------------------------------------------------------
    def __init__( self, 
                  p_mode, 
                  p_controllers : ControllerList,
                  p_controlled_systems : ControlledSystemList,
                  p_name : str = '',
                  p_cycle_limit = 0, 
                  p_visualize : bool = False, 
                  p_logging = Log.C_LOG_ALL,
                  **p_kwargs ):
                
        if ( len(p_controllers) == 0) or ( len(p_controllers) != len(p_controlled_systems) ):
            raise ParamError( 'Please provide an equal number of controllers and related controlled systems')
        
        self._controllers           = p_controllers.copy()
        self._controlled_systems    = p_controlled_systems.copy()
        self.set_name( p_name = p_name )

        super().__init__( p_mode = p_mode,
                          p_cycle_limit = p_cycle_limit,
                          p_visualize = p_visualize,
                          p_logging = p_logging,
                          **p_kwargs )
        

## -------------------------------------------------------------------------------------------------
    def _add_tasks_to_workflow(self, p_workflow : ControlWorkflow, p_tasks, p_pred_tasks = [] ) -> ControlTask:

        if isinstance( p_tasks, list):
            tasks = p_tasks
        else:
            tasks = [ p_tasks ]

        pred_tasks = p_pred_tasks

        for task in tasks:
            if isinstance( task, System ):
                # Native systems need to be wrapped
                task = ControlledSystem( p_system = task,
                                         p_name = task.get_name(),
                                         p_range_max = task.get_range(),
                                         p_visualize = self.get_visualization(),
                                         p_logging = self.get_log_level() )

            p_workflow.add_task( p_task = task, p_pred_tasks = pred_tasks )
            pred_tasks = [ task ]

        return task
        

## -------------------------------------------------------------------------------------------------
    def _setup(self, p_mode, p_visualize: bool, p_logging, **p_kwargs) -> ControlWorkflow:

        # 0 Intro
        workflow_prev : ControlWorkflow = None


        # 1 Create cascaded workflows from config data
        workflow_list = list(zip(self._controllers, self._controlled_systems))
        workflow_list.reverse()
        num_workflows = len( workflow_list )

        for i, (t_ctrl, t_ctrl_sys) in enumerate(workflow_list):

            workflow_id = num_workflows - i - 1

            workflow = ControlWorkflow( p_mode = p_mode,
                                        p_name = str(workflow_id),
                                        p_visualize = p_visualize,
                                        p_logging = p_logging )
            
            t_comp = Comparator( p_visualize = p_visualize, p_logging = p_logging )
            workflow.add_task( p_task = t_comp  )

            last_task = self._add_tasks_to_workflow( p_workflow = workflow, p_tasks = t_ctrl, p_pred_tasks = [ t_comp ] )

            if workflow_prev is not None:
                t_conv = Converter( p_src_type = ControlVariable,
                                    p_dst_type = SetPoint,
                                    p_visualize = p_visualize, 
                                    p_logging = p_logging )
                workflow.add_task( p_task = t_conv, p_pred_tasks = [ last_task ]  )

                workflow.add_task( p_task = workflow_prev, p_pred_tasks = [ t_conv ] )

                t_conv = Converter( p_src_type = ControlledVariable,
                                    p_dst_type = ControlVariable,
                                    p_visualize = p_visualize, 
                                    p_logging = p_logging )
                workflow.add_task( p_task = t_conv, p_pred_tasks = [ workflow_prev ] )

                self._add_tasks_to_workflow( p_workflow = workflow, p_tasks = t_ctrl_sys, p_pred_tasks = [ t_conv ] )

            else:
                self._add_tasks_to_workflow( p_workflow = workflow, p_tasks = t_ctrl_sys, p_pred_tasks = [ last_task ] )
            
            workflow_prev = workflow


        # 2 Return the outer control workflow
        return workflow
        
