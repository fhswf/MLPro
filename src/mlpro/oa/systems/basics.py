## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - The integrative middleware framework for standardized machine learning
## -- Package : mlpro.oa.systems
## -- Module  : basics.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2023-05-30  0.0.0     LSB      Creation
## -- 2023-05-31  0.1.0     LSB      Visualization
## -- 2023-05-31  0.1.1     LSB      cleaning
## -- 2023-05-31  0.1.2     LSB      Visualization bug fixed
## -- 2023-06-06  0.1.3     LSB      Renaming _wf and run methods with *_strans
## -- 2024-05-24  0.2.0     DA       Refactoring class PseudoTask
## --                                - constructor: changes on parameter p_wrap_method
## --                                - method _run(): changes on parameters
## -- 2024-06-10  0.2.1     LSB      Fixing for the refactoring on stream processing
## -- 2025-06-06  0.3.0     DA       Refactoring: p_inst -> p_instance/s
## -- 2025-07-18  0.4.0     DA       Refactoring
## -------------------------------------------------------------------------------------------------

"""
Ver. 0.4.0 (2025-07-18)

This module provides modules and template classes for adaptive systems and adaptive functions.
"""


import copy
from typing import Callable
from datetime import timedelta

from mlpro.bf import Log, ParamError, PlotSettings, Mode 
from mlpro.bf.plot import Figure 
from mlpro.bf.events import Event
from mlpro.bf.mt import *
from mlpro.bf.math import ESpace, MSpace
from mlpro.bf.systems import *
from mlpro.bf.streams import *
from mlpro.bf.ml import Model
from mlpro.bf.ml.systems import *

from mlpro.oa.streams import *



# Export list for public API
__all__ = [ 'PseudoTask',
            'OAFctSTrans',
            'OAFctSuccess',
            'OAFctBroken',
            'OASystem' ]




## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class PseudoTask(OAStreamTask):
    """
    A template class PseudoTask, only to be used by the OASystem. This functions runs a wrapped method as it's run
    method.

    Parameters
    ----------
    p_wrap_method
    p_name
    p_range_max
    p_duplicate_data
    p_logging
    p_visualize
    p_kwargs
    """

## -------------------------------------------------------------------------------------------------
    def __init__(self,
                 p_wrap_method:Callable[[InstDict],None],
                 p_name='PseudoTask',
                 p_range_max=Range.C_RANGE_NONE,
                 p_duplicate_data=True,
                 p_logging=Log.C_LOG_ALL,
                 p_visualize=False,
                 **p_kwargs):

        OAStreamTask.__init__(self,
                        p_name = p_name,
                        p_range_max = p_range_max,
                        p_duplicate_data = p_duplicate_data,
                        p_ada = False,
                        p_logging = p_logging,
                        p_visualize= p_visualize,
                        **p_kwargs)


        self._host_task = p_wrap_method


## -------------------------------------------------------------------------------------------------
    def _run( self, p_instances : InstDict ):

        self._host_task( p_instances = p_instances )





## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class OAFctSTrans(FctSTrans, Model):
    """
    This is a template class for Online Adaptive State Transition function. Please overwrite the
    _simulate_reaction() method or provide an adaptive class as a parameter with all the additional
    required parameters.

    Parameters
    ----------
    p_id
    p_name
    p_range_max
    p_autorun
    p_class_shared
    p_ada
    p_afct_cls
    p_state_space
    p_action_space
    p_input_space_cls
    p_output_space_cls
    p_output_elem_cls
    p_threshold
    p_buffer_size
    p_wf
    p_visualize
    p_logging
    p_kwargs
    """

## -------------------------------------------------------------------------------------------------
    def __init__(self,
                 p_id = None,
                 p_name: str = None,
                 p_range_max: int = Async.C_RANGE_PROCESS,
                 p_autorun: int = Task.C_AUTORUN_NONE,
                 p_class_shared = None,
                 p_ada:bool=True,
                 p_afct_cls = None,
                 p_state_space: MSpace = None,
                 p_action_space: MSpace = None,
                 p_input_space_cls=ESpace,
                 p_output_space_cls=ESpace,
                 p_output_elem_cls=State,  # Specific output element type
                 p_threshold=0,
                 p_buffer_size=0,
                 p_wf_strans: OAStreamWorkflow = None,
                 p_visualize:bool=False,
                 p_logging=Log.C_LOG_ALL,
                 **p_kwargs):

        self._afct_strans = None
        if p_afct_cls is not None:
            if (p_state_space is None) or (p_action_space is None):
                raise ParamError("Please provide mandatory parameters state and action space.")

            self._afct_strans = AFctSTrans(p_afct_cls = p_afct_cls,
                                          p_state_space=p_state_space,
                                          p_action_space=p_action_space,
                                          p_input_space_cls=p_input_space_cls,
                                          p_output_space_cls=p_output_space_cls,
                                          p_output_elem_cls=p_output_elem_cls,
                                          p_threshold=p_threshold,
                                          p_buffer_size=p_buffer_size,
                                          p_ada=p_ada,
                                          p_visualize=p_visualize,
                                          p_logging=p_logging,
                                          **p_kwargs)

        FctSTrans.__init__(self, p_logging = p_logging)

        Model.__init__(self,
                       p_id= p_id,
                       p_name=p_name,
                       p_range_max=p_range_max,
                       p_autorun=p_autorun,
                       p_class_shared=p_class_shared,
                       p_ada=p_ada,
                       p_visualize=p_visualize,
                       p_logging=p_logging,
                       **p_kwargs)

        if p_wf_strans is None:
            self._wf_strans = OAStreamWorkflow(p_name='State Transition',
                                  p_visualize=p_visualize,
                                  p_ada=p_ada,
                                  p_logging=p_logging)
        else:
            self._wf_strans = p_wf_strans

        self._action_obj:Action = None
        self._setup_wf_strans = False
        self._state_id = 0


## -------------------------------------------------------------------------------------------------
    def simulate_reaction(self, p_state: State, p_action: Action, p_t_step : timedelta = None) -> State:
        """
        Simulates a state transition based on a state and action. Custom method _simulate_reaction()
        is called.

        Parameters
        ----------
        p_state: State
            State of the System.

        p_action: Action
            External action provided for the action simulation

        p_t_step: timedelta, optional.
            The timestep for which the system is to be simulated

        Returns
        -------
        state: State
            The new state of the System.

        """

        self._state_obj = p_state.copy()
        self._action_obj = copy.deepcopy(p_action)
        self.log(Log.C_LOG_TYPE_I, 'Reaction Simulation Started...')

        # 2. Checking for the first run
        if not self._setup_wf_strans:
            self._setup_wf_strans = self._setup_oafct_strans()

        # 3. Running the workflow
        self._wf_strans.run( p_instances = dict([(self._state_obj.get_id(), (InstTypeNew, self._state_obj))]) )


        # 4. get the results
        state = self._wf_strans.get_so().get_results()[self.get_id()]

        # state.set_id(self._state_id)
        state.set_id(self._state_id)
        self._state_id += 1

        return state


## -------------------------------------------------------------------------------------------------
    def _adapt(self, **p_kwargs) -> bool:
        """
        When called, the function and it's components adapt based on the provided parameters.

        Parameters
        ----------
        p_kwargs
            additional parameters for adaptation.

        Returns
        -------
        adapted: bool
            Returns true if the Function has adapted
        """

        adapted = False
        try:
            adapted = self._wf_strans.adapt(**p_kwargs) or adapted
        except:
            adapted = adapted or False

        if self._afct_strans is not None:
            try:
                adapted = self._afct_strans.adapt(**p_kwargs) or adapted
            except:
                adapted = adapted or False

        return adapted


## -------------------------------------------------------------------------------------------------
    def _adapt_on_event(self, p_event_id:str, p_event_object:Event) -> bool:
        """

        Parameters
        ----------
        p_event_id
        p_event_object

        Returns
        -------

        """
        pass


## -------------------------------------------------------------------------------------------------
    def add_task_strans(self, p_task:OAStreamTask, p_pred_task = None):
        """
        Adds a task to the workflow.

        Parameters
        ----------
        p_task: OATask, StreamTask
            The task to be added to the workflow

        p_pred_task: list[Task]
            Name of the predecessor tasks for the task to be added

        """
        self._wf_strans.add_task(p_task, p_pred_task)


## -------------------------------------------------------------------------------------------------
    def _run_wf_strans(self, p_instances:InstDict):
        """
        Runs the processing workflow, for state transition.

        Parameters
        ----------
        p_instances : InstDict
            Dictionary of instances to be processed by the workflow.

        """
        inst_new = [inst[1] for inst in p_instances.values() if inst[0] == InstTypeNew]

        if self._afct_strans is not None:
            self._wf_strans.get_so().add_result(self.get_id(), AFctSTrans.simulate_reaction(self._afct_strans,
                                                                                p_state=inst_new[0],
                                                                                p_action=self._action_obj))
        else:
            self._wf_strans.get_so().add_result(self.get_id(), FctSTrans.simulate_reaction(self,
                                                                            p_state=inst_new[0],
                                                                            p_action=self._action_obj))


## -------------------------------------------------------------------------------------------------
    def _setup_oafct_strans(self):
        """
        Adds a pseudo task to the processing workflow, with the method to be wrapped.

        Returns
        -------
        bool
            False when successfully setup.

        """

        if len(self._wf_strans._tasks) == 0:
            p_pred_tasks = None
        else:
            p_pred_tasks = [self._wf_strans._tasks[-1]]
            self._wf_strans = OAStreamWorkflow()
        self._wf_strans.add_task(p_task=PseudoTask(p_wrap_method = self._run_wf_strans, p_logging=self.get_log_level()),
                                 p_pred_tasks=p_pred_tasks)

        return True





## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class OAFctSuccess(FctSuccess, Model):
    """
    This is a template class for Online Adaptive Success Computation function. Please overwrite the
    _compute_success() method or provide an adaptive class as a parameter with all the additional
    required parameters.

    Parameters
    ----------
    p_id
    p_name
    p_range_max
    p_autorun
    p_class_shared
    p_ada
    p_afct_cls
    p_state_space
    p_action_space
    p_input_space_cls
    p_output_space_cls
    p_output_elem_cls
    p_threshold
    p_buffer_size
    p_wf_success
    p_visualize
    p_logging
    p_kwargs
    """

## -------------------------------------------------------------------------------------------------
    def __init__(self,
                 p_id = None,
                 p_name: str = None,
                 p_range_max: int = Async.C_RANGE_PROCESS,
                 p_autorun: int = Task.C_AUTORUN_NONE,
                 p_class_shared = None,
                 p_ada:bool=True,
                 p_afct_cls = None,
                 p_state_space: MSpace = None,
                 p_action_space: MSpace = None,
                 p_input_space_cls=ESpace,
                 p_output_space_cls=ESpace,
                 p_output_elem_cls=State,  # Specific output element type
                 p_threshold=0,
                 p_buffer_size=0,
                 p_wf_success: OAStreamWorkflow = None,
                 p_visualize:bool=False,
                 p_logging=Log.C_LOG_ALL,
                 **p_kwargs):

        self._afct_success = None
        if p_afct_cls is not None:
            if (p_state_space is None) or (p_action_space is None):
                raise ParamError("Please provide mandatory parameters state and action space.")

            self._afct_success = AFctSuccess(p_afct_cls=p_afct_cls,
                                            p_state_space=p_state_space,
                                            p_action_space=p_action_space,
                                            p_input_space_cls=p_input_space_cls,
                                            p_output_space_cls=p_output_space_cls,
                                            p_output_elem_cls=p_output_elem_cls,
                                            p_threshold=p_threshold,
                                            p_buffer_size=p_buffer_size,
                                            p_ada=p_ada,
                                            p_visualize=p_visualize,
                                            p_logging=p_logging,
                                            **p_kwargs)

        FctSuccess.__init__(self, p_logging=p_logging)

        Model.__init__(self,
                       p_id= p_id,
                       p_name=p_name,
                       p_range_max=p_range_max,
                       p_autorun=p_autorun,
                       p_class_shared=p_class_shared,
                       p_ada=p_ada,
                       p_visualize=p_visualize,
                       p_logging=p_logging,
                       **p_kwargs)

        if p_wf_success is None:
            self._wf_success = OAStreamWorkflow(p_name='Success Computation',
                                          p_visualize=p_visualize,
                                          p_ada=p_ada,
                                          p_logging=p_logging)
        else:
            self._wf_success = p_wf_success

        self._setup_wf_success = False


## -------------------------------------------------------------------------------------------------
    def compute_success(self, p_state: State) -> bool:
        """
        Assesses the given state regarding success criteria. Custom method _compute_success() is called.

        Parameters
        ----------
        p_state : State
            System state.

        Returns
        -------
        success : bool
            True, if given state is a success state. False otherwise.
        """
        # 1. Create copy of the state parameter
        self._state_obj = p_state.copy()
        self.log(Log.C_LOG_TYPE_I, 'Assessing Success...')

        # 2. Set up the Success workflow, if not already
        if not self._setup_wf_success:
            self._setup_wf_success = self._setup_oafct_success()

        # 3. Run the workflow
        self._wf_success.run( p_instances = dict([(self._state_obj.get_id(), (InstTypeNew, self._state_obj))]) )


        # 4. Return the results
        return self._wf_success.get_so().get_results()[self.get_id()]


## -------------------------------------------------------------------------------------------------
    def _adapt_on_event(self, p_event_id:str, p_event_object:Event) -> bool:
        """

        Parameters
        ----------
        p_event_id
        p_event_object

        Returns
        -------

        """
        pass


## -------------------------------------------------------------------------------------------------
    def add_task_success(self, p_task:StreamTask, p_pred_tasks:list = None):
        """
        Adds a task to the workflow.

        Parameters
        ----------
        p_task: OATask, StreamTask
            The task to be added to the workflow

        p_pred_task: list[Task]
            Name of the predecessor tasks for the task to be added

        """
        self._wf_success.add_task(p_task = p_task, p_pred_tasks=p_pred_tasks)


## -------------------------------------------------------------------------------------------------
    def _run_wf_success(self, p_instances : InstDict):
        """
        Runs the success computation workflow of the system.

        Parameters
        ----------
        p_instances : InstDict
            Dictionary of instances to be processed by the workflow.

        """
        inst_new = [inst[1] for inst in p_instances.values() if inst[0] == InstTypeNew]
        if self._afct_success is not None:
            self._wf_success.get_so().add_result(self.get_id(), AFctSuccess.compute_success(self._afct_success,
                                                                                 p_state=inst_new[0]))
        else:
            self._wf_success.get_so().add_result(self.get_id(), FctSuccess.compute_success(self,
                                                                            p_state=inst_new[0]))


## -------------------------------------------------------------------------------------------------
    def _adapt(self, **p_kwargs) -> bool:
        """
        When called, the function and it's components adapt based on the provided parameters.

        Parameters
        ----------
        p_kwargs
            additional parameters for adaptation.

        Returns
        -------
        adapted: bool
            Returns true if the Function has adapted
        """

        adapted = False
        try:
            adapted = self._wf_success.adapt(**p_kwargs) or adapted
        except:
            adapted = False or adapted

        if self._afct_success is not None:
            try:
                adapted = self._afct_success.adapt(**p_kwargs) or adapted
            except:
                adapted = False or adapted

        return adapted


## -------------------------------------------------------------------------------------------------
    def _setup_oafct_success(self):
        """
        Adds a pseudo task to the success computation workflow, with the method to be wrapped.

        Returns
        -------
        bool
            False when successfully setup.

        """
        if len(self._wf_success._tasks) == 0:
            p_pred_tasks = None
        else:
            p_pred_tasks = [self._wf_success._tasks[-1]]

        self._wf_success.add_task(p_task = PseudoTask(p_wrap_method = self._run_wf_success, p_logging=self.get_log_level()),
                                  p_pred_tasks=p_pred_tasks)
        return True





## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class OAFctBroken(FctBroken, Model):
    """
    This is a template class for Online Adaptive Broken Computation function. Please
    overwrite the _compute_broken() method or provide an adaptive class as a parameter with all the
    additional required parameters.

    Parameters
    ----------
    p_id
    p_name
    p_range_max
    p_autorun
    p_class_shared
    p_ada
    p_afct_cls
    p_state_space
    p_action_space
    p_input_space_cls
    p_output_space_cls
    p_output_elem_cls
    p_threshold
    p_buffer_size
    p_wf_broken
    p_visualize
    p_logging
    p_kwargs
    """

## -------------------------------------------------------------------------------------------------
    def __init__(self,
                 p_id = None,
                 p_name: str = None,
                 p_range_max: int = Async.C_RANGE_PROCESS,
                 p_autorun: int = Task.C_AUTORUN_NONE,
                 p_class_shared = None,
                 p_ada:bool=True,
                 p_afct_cls = None,
                 p_state_space: MSpace = None,
                 p_action_space: MSpace = None,
                 p_input_space_cls=ESpace,
                 p_output_space_cls=ESpace,
                 p_output_elem_cls=State,  # Specific output element type
                 p_threshold=0,
                 p_buffer_size=0,
                 p_wf_broken: OAStreamWorkflow = None,
                 p_visualize:bool=False,
                 p_logging=Log.C_LOG_ALL,
                 **p_kwargs):

        self._afct_broken = None
        if p_afct_cls is not None:
            if (p_state_space is None) or (p_action_space is None):
                raise ParamError("Please provide mandatory parameters state and action space.")


            self._afct_broken = AFctBroken(p_afct_cls=p_afct_cls,
                                            p_state_space=p_state_space,
                                            p_action_space=p_action_space,
                                            p_input_space_cls=p_input_space_cls,
                                            p_output_space_cls=p_output_space_cls,
                                            p_output_elem_cls=p_output_elem_cls,
                                            p_threshold=p_threshold,
                                            p_buffer_size=p_buffer_size,
                                            p_ada=p_ada,
                                            p_visualize=p_visualize,
                                            p_logging=p_logging,
                                            **p_kwargs)
        #
        # else:
        FctBroken.__init__(self, p_logging=p_logging)

        Model.__init__(self,
                       p_id= p_id,
                       p_name=p_name,
                       p_range_max=p_range_max,
                       p_autorun=p_autorun,
                       p_class_shared=p_class_shared,
                       p_ada=p_ada,
                       p_visualize=p_visualize,
                       p_logging=p_logging,
                       **p_kwargs)

        if p_wf_broken is None:
            self._wf_broken = OAStreamWorkflow(p_name='Broken Computation',
                                         p_visualize=p_visualize,
                                         p_ada=p_ada,
                                         p_logging=p_logging)
        else:
            self._wf_broken = p_wf_broken

        self._setup_wf_broken = False


## -------------------------------------------------------------------------------------------------
    def compute_broken(self, p_state: State) -> bool:
        """
        Assesses the given state regarding breakdown criteria. Custom method _compute_success() is called.

        Parameters
        ----------
        p_state : State
            System state.

        Returns
        -------
        broken : bool
            True, if given state is a breakdown state. False otherwise.
        """
        # 1. Create a copy of the state parameter
        self._state_obj = p_state.copy()
        self.log(Log.C_LOG_TYPE_I, 'Assessing Broken...')

        # 2. Set up the broken computation workflow, if not already
        if not self._setup_wf_broken:
            self._setup_wf_broken = self._setup_oafct_broken()

        # 3. Run the workflow
        self._wf_broken.run( p_instances = dict([(self._state_obj.get_id(), (InstTypeNew, self._state_obj))]) )


        # 4. Return the results
        return self._wf_broken.get_so().get_results()[self.get_id()]


## -------------------------------------------------------------------------------------------------
    def add_task_broken(self, p_task: StreamTask, p_pred_tasks:list = None):
        """
        Adds a task to the workflow.

        Parameters
        ----------
        p_task: OATask, StreamTask
            The task to be added to the workflow

        p_pred_task: list[Task]
            Name of the predecessor tasks for the task to be added

        """
        self._wf_broken.add_task(p_task = p_task, p_pred_tasks = p_pred_tasks)


## -------------------------------------------------------------------------------------------------
    def _run_wf_broken(self, p_instances : InstDict):
        """
        Runs the success computation workflow of the system.

        Parameters
        ----------
        p_instances : InstDict
            Dictionary of instances to be processed by the workflow.
        """
        
        inst_new = [inst[1] for inst in p_instances.values() if inst[0] == InstTypeNew]

        if self._afct_broken is not None:
            self._wf_broken.get_so().add_result(self.get_id(), AFctBroken.compute_broken(self._afct_broken,
                                                                         p_state=inst_new[0]))
        else:
            self._wf_broken.get_so().add_result(self.get_id(), FctBroken.compute_broken(self,
                                                                         p_state=inst_new[0]))


## -------------------------------------------------------------------------------------------------
    def _adapt(self, **p_kwargs) -> bool:
        """
        Custom method to adapt all the internal adaptive elements of the OAFct Broken.

        Parameters
        ----------
        p_kwargs:
            Parameters to be used for adaptation.

        Returns
        -------
        bool
            True if any of the element has adapted.
        """

        adapted = False
        try:
            adapted = self._wf_broken.adapt(**p_kwargs) or adapted
        except:
            adapted = False or adapted

        if self._afct_broken is not None:
            try:
                adapted = self._afct_broken.adapt(**p_kwargs) or adapted
            except:
                adapted = False or adapted

        return adapted


## -------------------------------------------------------------------------------------------------
    def _adapt_on_event(self, p_event_id:str, p_event_object:Event) -> bool:
        """

        Parameters
        ----------
        p_event_id
        p_event_object

        Returns
        -------

        """
        pass


## -------------------------------------------------------------------------------------------------
    def _setup_oafct_broken(self):
        """
        Adds the PseudoTask with broken computation function as the host task to the workflow

        Returns
        -------
        bool
            True after the setup is completed.

        """

        if len(self._wf_broken._tasks) == 0:
            p_pred_tasks = None
        else:
            p_pred_tasks = [self._wf_broken._tasks[-1]]
        self._wf_broken.add_task(p_task=PseudoTask(p_wrap_method = self._run_wf_broken, p_logging=self.get_log_level()),
                                 p_pred_tasks=p_pred_tasks)
        return True





## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class OASystem(OAFctBroken, OAFctSTrans, OAFctSuccess, ASystem):
    """
    This ist a template class for Adaptive State Based System.

    Parameters
    ----------
    p_id
        Id of the system.
    p_name:str
        Name of the system.
    p_range_max
        Range of the system.
    p_autorun
        Whether the system should autorun as a Task.
    p_class_shared
        The shared class for multisystem.
    p_mode
        Mode of the System. Simulation or real.
    p_ada:bool
        The adaptability of the system.
    p_latency:timedelta
        Latency of the system.
    p_t_step:timedelta
        Simulation timestep of the system.
    p_fct_strans: FctSTrans | AFctSTrans | OAFctSTrans
        External state transition function.
    p_fct_success: FctSuccess | AFctSuccess | OAFctSuccess
        External success computation function.
    p_fct_broken: FctBroken | AFctBroken | OAFctBroken
        External broken computation function.
    p_wf_strans: OAWorkflow
        State transition workflow. Optional.
    p_wf_success: OAWorkflow
        Success computation workflow. Optional.
    p_wf_broken: OAWorkflow
        Broken computation workflow. Optional
    p_mujoco_file
        Mujoco file for simulation using mujoco engine.
    p_frame_skip
        Number of frames to be skipped during visualization.
    p_state_mapping:
        State mapping for Mujoco.
    p_action_mapping:
        Action Mapping for Mujoco.
    p_camera_conf:
        Camera Configuration for Mujoco.
    p_visualize:
        Visualization switch.
    p_logging
        Logging level for the system.
    p_kwargs
        Additional Parameters
    """


## -------------------------------------------------------------------------------------------------
    def __init__(self,
                 p_id=None,
                 p_name: str = None,
                 p_range_max: int = Range.C_RANGE_NONE,
                 p_autorun: int = Task.C_AUTORUN_NONE,
                 p_class_shared: Shared = None,
                 p_ada:bool = True,
                 p_mode = Mode.C_MODE_SIM,
                 p_latency = None,
                 p_t_step: timedelta = None,
                 p_fct_strans : FctSTrans = None,
                 p_fct_success : FctSuccess = None,
                 p_fct_broken : FctBroken = None,
                 p_wf_strans : OAStreamWorkflow = None,
                 p_wf_success : OAStreamWorkflow = None,
                 p_wf_broken : OAStreamWorkflow = None,
                 p_mujoco_file = None,
                 p_frame_skip: int = 1,
                 p_state_mapping = None,
                 p_action_mapping = None,
                 p_camera_conf: tuple = (None, None, None),
                 p_visualize: bool = False,
                 p_logging: bool = Log.C_LOG_ALL,
                 **p_kwargs):

        self._workflows = []
        self._fcts =[]

        OAFctSTrans.__init__(self, p_name=p_name, p_wf_strans=p_wf_strans, p_visualize=p_visualize, p_logging=p_logging)

        OAFctSuccess.__init__(self, p_name=p_name, p_wf=p_wf_success, p_visualize=p_visualize, p_logging=p_logging)

        OAFctBroken.__init__(self, p_name=p_name, p_wf_broken=p_wf_broken, p_visualize=p_visualize, p_logging=p_logging)

        self._workflows = [self._wf_strans, self._wf_success, self._wf_broken]

        ASystem.__init__(self,
                         p_id = p_id,
                         p_name = p_name,
                         p_range_max = p_range_max,
                         p_autorun = p_autorun,
                         p_class_shared = p_class_shared,
                         p_mode = p_mode,
                         p_ada = p_ada,
                         p_latency = p_latency,
                         p_t_step = p_t_step,
                         p_fct_strans = p_fct_strans,
                         p_fct_success = p_fct_success,
                         p_fct_broken = p_fct_broken,
                         p_mujoco_file = p_mujoco_file,
                         p_frame_skip = p_frame_skip,
                         p_state_mapping = p_state_mapping,
                         p_action_mapping = p_action_mapping,
                         p_camera_conf = p_camera_conf,
                         p_visualize = p_visualize,
                         p_logging = p_logging,
                         **p_kwargs)



## -------------------------------------------------------------------------------------------------
    def _adapt(self, **p_kwargs) -> bool:
        """
        Adapts the system based on the parameters provided. Calls the adapt methods of all the adaptive
        elements of the system.

        Parameters
        ----------
        p_kwargs
            Additional parameters for the adaptation of the system.

        Returns
        -------
        bool
            True if any of the element has adapted.

        """

        adapted = False

        for workflow in self._workflows:
            try:
                adapted = workflow.adapt(**p_kwargs) or adapted
            except:
                pass

        for fct in self._fcts:
            try:
                adapted = fct.adapt(**p_kwargs) or adapted
            except:
                pass


        return adapted


## -------------------------------------------------------------------------------------------------
    def switch_adaptivity(self, p_ada:bool):
        """
        Switches the adaptivity of all the internal adaptive elements of OASystem.

        Parameters
        ----------
        p_ada: bool
            The boolean flag indicating if the adaptivity shall be switched on or off.

        """
        for workflow in self._workflows:
            try:
                workflow.switch_adaptivity(p_ada=p_ada)
            except:
                pass

        for fct in self._fcts:
            try:
                fct.switch_adaptivity(p_ada = p_ada)
            except:
                pass

        Model.switch_adaptivity(self, p_ada=p_ada)

## -------------------------------------------------------------------------------------------------
    def _adapt_on_event(self, p_event_id:str, p_event_object:Event) -> bool:
        """

        Parameters
        ----------
        p_event_id
        p_event_object

        Returns
        -------

        """
        pass


## -------------------------------------------------------------------------------------------------
    def simulate_reaction(self, p_state: State, p_action: Action, p_t_step : timedelta = None) -> State:
        """
        Simulates a state transition based on a state and action. Custom method _simulate_reaction()
        is called.

        Parameters
        ----------
        p_state: State
            State of the System.

        p_action: Action
            External action provided for the action simulation

        p_t_step: timedelta, optional.
            The timestep for which the system is to be simulated

        Returns
        -------
        state: State
            The new state of the System.

        """
        # 1. copying the state and action object for the function level
        try:
            p_state.get_id()
        except:
            p_state.set_id(self._state_id)
            self._state_id += 1

        if self._fct_strans is not None:
            state = self._fct_strans.simulate_reaction(p_state, p_action, p_t_step)
        else:
            state = OAFctSTrans.simulate_reaction(self, p_state, p_action, p_t_step)

        try:
            state.get_id()
        except:
            state.set_id(self._state_id)
            self._state_id += 1

        return state

## -------------------------------------------------------------------------------------------------
    def compute_success(self, p_state: State) -> bool:
        """
        Assesses the given state regarding success criteria. Custom method _compute_success() is called.

        Parameters
        ----------
        p_state : State
            System state.

        Returns
        -------
        success : bool
            True, if given state is a success state. False otherwise.

        """

        if self._fct_success is not None:
            return self._fct_success.compute_success(p_state)
        else:
            return OAFctSuccess.compute_success(self, p_state)


## -------------------------------------------------------------------------------------------------
    def compute_broken(self, p_state: State) -> bool:
        """
        Assesses the given state regarding breakdown criteria. Custom method _compute_success() is called.

        Parameters
        ----------
        p_state : State
            System state.

        Returns
        -------
        broken : bool
            True, if given state is a breakdown state. False otherwise.
        """
        if self._fct_broken is not None:
            return self._fct_broken.compute_broken(p_state)
        else:
            return OAFctBroken.compute_broken(self, p_state)


## -------------------------------------------------------------------------------------------------
    def init_plot(self,
                      p_figure: Figure = None,
                      p_plot_settings: PlotSettings = None,
                      **p_kwargs):
        """
        Initializes the plot for all the internal elements of OASystem.

        Parameters
        ----------
        p_figure: Figure, optional
            Matplotlib figure, if one exists already

        p_plot_settings: PlotSettings
            Additional plot settings

        p_kwargs:
            Additional plot parameters

        """

        super().init_plot(p_figure=p_figure, p_plot_settings=p_plot_settings, **p_kwargs)

        for fct in self._fcts:
            try:
                fct.init_plot(p_figure=p_figure, p_plot_settings=p_plot_settings, **p_kwargs)
            except:
                pass

        for workflow in self._workflows:
            try:
                workflow.init_plot(p_figure=p_figure, p_plot_settings=p_plot_settings)
            except:
                pass


## -------------------------------------------------------------------------------------------------
    def update_plot(self, **p_kwargs):
        """
        Updates the all the internal plots.

        Parameters
        ----------
        p_kwargs: additional plot parameters

        """
        super().update_plot(**p_kwargs)

        for fct in self._fcts:
            try:
                fct.update_plot(**p_kwargs)
            except:
                pass

        # for workflow in self._workflows:
        #     try:
        #         workflow.update_plot(**p_kwargs)
        #     except:
        #         pass


## -------------------------------------------------------------------------------------------------
    def switch_visualization(self, p_object = None, p_visualize = None):
        """
        Method to switch the visualization of an object in an online adaptive system.

        Parameters
        ----------
        p_object: object
            The object whose visualization is to be switched off. Although, a valid object is any object
            with visualization property in MLPro, within this runtime. It is suggested to use only on the Functions,
            Workflows, Tasks and System itself.

        p_visualize: bool
            The bool value for setting the visualization of an object.

        Notes
        -----
        Please do not turn off the visualization by getting functions of the system (e.g. self.get_fctstrans()), if the functions are not
        provided externally to the system, since this will refer to the system itself.

        Examples
        --------
        myOASystem.switch_visualization(p_visualize = False, p_object = self.get_fct_workflow())

        """

        p_object._visualize = p_visualize


## -------------------------------------------------------------------------------------------------
    def get_workflow_strans(self):

        return self._wf_strans

## -------------------------------------------------------------------------------------------------
    def get_workflow_success(self):

        return self._wf_success


## -------------------------------------------------------------------------------------------------
    def get_workflow_broken(self):

        return self._wf_broken

