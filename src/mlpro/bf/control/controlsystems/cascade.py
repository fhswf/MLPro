## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - The integrative middleware framework for standardized machine learning
## -- Package : mlpro.bf.control.controlsystems
## -- Module  : cascade.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2024-11-08  1.0.0     DA       Initial implementation
## -------------------------------------------------------------------------------------------------

"""
Ver. 1.0.0 (2024-11-08)

This module provides a container class for cascade control systems.

"""

from typing import List, Union

from mlpro.bf.various import Log
from mlpro.bf.exceptions import *
from mlpro.bf.control import *
from mlpro.bf.control.operators import Comparator, Converter


ControllerList = List[ Union[ Controller, List[ControlTask] ] ]
ControlledSystemList = List[ Union[ ControlledSystem, List[ControlTask] ] ]


## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class CascadeControlSystem (ControlSystem):
    """
    Cascade control system.

    Parameters
    ----------
    p_controllers : ControllerList
        List of controllers to be cascaded in order to outer to inner controller.
    p_controlled_systems : ControlledSystemLists
        List of controlled systems to be cascaded in order to outer to inner controlled system.
    p_ctrl_var_integration : bool = False
        If True, an optional intrator is added to control workflow
    """

    C_TYPE          = 'Cascade Control System'

## -------------------------------------------------------------------------------------------------
    def __init__( self, 
                  p_controllers : ControllerList,
                  p_controlled_systems : ControlledSystemList,
                  p_mode, 
                  p_cycle_limit = 0, 
                  p_visualize : bool = False, 
                  p_logging = Log.C_LOG_ALL ):
                
        if ( len(p_controllers) == 0) or ( len(p_controllers) != len(p_controlled_systems) ):
            raise ParamError( 'Please provide an equal number of controllers and related controlled systems')
        
        self._controllers           = p_controllers.copy()
        self._controlled_systems    = p_controlled_systems.copy()

        super().__init__( p_mode = p_mode,
                          p_cycle_limit = p_cycle_limit,
                          p_visualize = p_visualize,
                          p_logging = p_logging )
        

## -------------------------------------------------------------------------------------------------
    def _add_tasks_to_workflow(self, p_workflow : ControlWorkflow, p_tasks, p_pred_tasks = [] ) -> ControlTask:

        if isinstance( p_tasks, list):
            tasks = p_tasks
        else:
            tasks = [ p_tasks ]

        pred_tasks = p_pred_tasks

        for task in tasks:
            p_workflow.add_task( p_task = task, p_pred_tasks = pred_tasks )
            pred_tasks = [ task ]

        return task
        

## -------------------------------------------------------------------------------------------------
    def _setup(self, p_mode, p_visualize: bool, p_logging) -> ControlWorkflow:

        workflow_prev : ControlWorkflow = None

        for i, t_ctrl, t_ctrl_sys in enumerate(list(zip(self._controllers, self._controlled_systems)).reverse()):

            workflow = ControlWorkflow( p_mode = p_mode,
                                        p_name = str(i),
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

        return workflow
        
