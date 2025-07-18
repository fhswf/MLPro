## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - The integrative middleware framework for standardized machine learning
## -- Package : mlpro.rl
## -- Module  : models_adaptive_environment.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2023-05-30  0.0.0     LSB      Creation
## -- 2023-05-31  0.1.0     LSB      Visulization
## -- 2023-05-31  0.1.1     LSB      Cleaning
## -- 2023-05-31  0.1.2     LSB      Visualization fixed
## -- 2023-06-10  0.1.3     LSB      Fixed for refactoring on stream processing
## -- 2025-06-06  0.2.0     DA       Refactoring: p_inst -> p_instance/s
## -- 2025-07-17  0.3.0     DA       Refactoring
## -------------------------------------------------------------------------------------------------

"""
Ver. 0.3.0 (2025-07-17) 

This module provides model classes for adaptive environments
"""

from datetime import timedelta

from mlpro.bf import Log, ParamError, Mode
from mlpro.bf.events import Event
from mlpro.bf.mt import *
from mlpro.bf.math import MSpace, ESpace
from mlpro.bf.ml import Model
from mlpro.bf.systems import State, FctSTrans, FctSuccess, FctBroken
from mlpro.bf.streams import InstTypeNew, InstDict, StreamTask

from mlpro.oa.streams import OAStreamWorkflow
from mlpro.oa.systems import *
from mlpro.rl import *



# Export list for public API
__all__ = [ 'OAFctReward',
            'OAEnvironment' ]




## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class OAFctReward(FctReward, Model):
    """
    This is a template class for Online Adaptive Reward Computation function. Please overwrite the
    _compute_reward() method or provide an adaptive class as a parameter with all the additional
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
    p_wf_reward
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
                 p_wf_reward: OAStreamWorkflow = None,
                 p_visualize:bool=False,
                 p_logging=Log.C_LOG_ALL,
                 **p_kwargs):

        self._afct_reward = None
        if p_afct_cls is not None:
            if (p_state_space is None) or (p_action_space is None):
                raise ParamError("Please provide mandatory parameters state and action space.")

            self._afct_reward = AFctReward(p_afct_cls=p_afct_cls,
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

        FctReward.__init__(self, p_logging = p_logging)

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

        if p_wf_reward is None:
            self._wf_reward = OAStreamWorkflow(p_name='Reward Computation',
                                         p_visualize=p_visualize,
                                         p_ada=p_ada,
                                         p_logging=p_logging)
        else:
            self._wf_reward = p_wf_reward


        self._state_obj_old:State = None
        self._state_obj_new:State = None
        self._setup_wf_reward = False
        self._inst_old:State = None


## -------------------------------------------------------------------------------------------------
    def compute_reward(self, p_state_old: State = None, p_state_new: State = None) -> Reward:
        """

        Parameters
        ----------
        p_state
        p_state_new

        Returns
        -------

        """

        # 1. Copy the state parameters for further processing
        self.log(Log.C_LOG_TYPE_I, 'Start Computing the Reward....')
        if p_state_old is not None:
            self._state_obj_old = p_state_old.copy()
        self._state_obj_new = p_state_new.copy()

        # 2. Check for the first run
        if not self._setup_wf_reward:
            self._setup_wf_reward = self._setup_oafct_reward()

        # 2.1 Check if the first instance (old state) is already processed
        if self._inst_old is None:
            inst_new = [self._state_obj_old, self._state_obj_new]
        else:
            inst_new = [self._state_obj_new]

        # 3. Run the workflow
        self._wf_reward.run( p_instances = dict(zip([inst.get_id() for inst in inst_new], [(InstTypeNew,inst) for inst in inst_new])) )

        # 4. Return the results
        return self._wf_reward.get_so().get_results()[self.get_tid()]


## -------------------------------------------------------------------------------------------------
    def _adapt_on_event(self, p_event_id: str, p_event_object: Event) -> bool:
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
    def add_task_reward(self, p_task: StreamTask, p_pred_tasks: list = None):
        """
        Adds a task to the workflow.
        Parameters
        ----------
        p_task: OATask, StreamTask
            The task to be added to the workflow

        p_pred_tasks: list[Task]
            Name of the predecessor tasks for the task to be added

        """

        self._wf_reward.add_task(p_task=p_task, p_pred_tasks=p_pred_tasks)


## -------------------------------------------------------------------------------------------------
    def _run_wf_reward(self, p_instances : InstDict):
        """
        Runs the reward computation workflow of the system.

        Parameters
        ----------
        p_instances : InstDict
            Instances to be processed.

        """
        ids = sorted(p_instances.keys())
        if len(ids) > 1:
            instances =  [p_instances[ids[0]][1], p_instances[ids[1]][1]]
        else:
            instances = [p_instances[ids[0]][1]]
        if len(instances) == 2 :
            state_new = instances[1]
            self._inst_old = instances[0]

        elif len(instances) == 1:
            state_new = instances[0]

        if self._inst_old is not None:
            state_old = self._inst_old

        else:
            state_old = instances[0]


        if self._afct_reward is not None:
            self._wf_reward.get_so().add_result(self.get_id(), AFctReward.compute_reward(self,
                                                                              p_state_new=state_new,
                                                                              p_state_old=state_old))
        else:
            self._wf_reward.get_so().add_result(self.get_id(), FctReward.compute_reward(self,
                                                                             p_state_new=state_new,
                                                                             p_state_old=state_old))


# -------------------------------------------------------------------------------------------------
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
            adapted = self._wf_reward.adapt(**p_kwargs) or adapted
        except:
            adapted = False or adapted

        if self._afct_reward is not None:
            try:
                adapted = self._afct_reward.adapt(**p_kwargs) or adapted
            except:
                adapted = False or adapted

        return adapted


## -------------------------------------------------------------------------------------------------
    def _setup_oafct_reward(self):
        """
        Adds a pseudo task to the success computation workflow, with the method to be wrapped.

        Returns
        -------
        bool
            False when successfully setup.

        """
        if len(self._wf_reward._tasks) == 0:
            p_pred_tasks = None
        else:
            p_pred_tasks = [self._wf_reward._tasks[-1]]

        self._wf_reward.add_task(p_task=PseudoTask(p_wrap_method=self._run_wf_reward),
                                  p_pred_tasks=p_pred_tasks)
        return True





## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class OAEnvironment(OAFctReward, OASystem, Environment):
    """
    Template class for Online Adaptive Environments, which adds the Online Adaptive Reward Computation functionality
    over OASystem.

    Parameters
    ----------
    p_id
    p_name
    p_buffer_size
    p_ada
    p_range_max
    p_autorun
    p_class_shared
    p_mode
    p_latency
    p_t_step
    p_fct_strans
    p_fct_reward
    p_fct_success
    p_fct_broken
    p_wf
    p_wf_success
    p_wf_broken
    p_wf_reward
    p_mujoco_file
    p_frame_skip
    p_state_mapping
    p_action_mapping
    p_camera_conf
    p_visualize
    p_logging
    p_kwargs
    """


    ## -------------------------------------------------------------------------------------------------
    def __init__(self,
                 p_id = None,
                 p_name: str = None,
                 p_buffer_size: int = 0,
                 p_ada: bool = True,
                 p_range_max: int = Range.C_RANGE_NONE,
                 p_autorun: int = Task.C_AUTORUN_NONE,
                 p_class_shared: Shared = None,
                 p_mode=Mode.C_MODE_SIM,
                 p_latency: timedelta = None,
                 p_t_step: timedelta = None,
                 p_fct_strans: FctSTrans = None,
                 p_fct_reward: FctReward = None,
                 p_fct_success: FctSuccess = None,
                 p_fct_broken: FctBroken = None,
                 p_wf : OAStreamWorkflow = None,
                 p_wf_success : OAStreamWorkflow = None,
                 p_wf_broken : OAStreamWorkflow = None,
                 p_wf_reward : OAStreamWorkflow = None,
                 p_mujoco_file = None,
                 p_frame_skip: int = 1,
                 p_state_mapping = None,
                 p_action_mapping = None,
                 p_camera_conf: tuple = (None, None, None),
                 p_visualize: bool = False,
                 p_logging: bool = Log.C_LOG_ALL,
                 **p_kwargs):


        OAFctReward.__init__(self, p_wf_reward=p_wf_reward, p_visualize=p_visualize)

        Environment.__init__(self,
                             p_mode = p_mode,
                             p_latency = p_latency,
                             p_fct_strans = p_fct_strans,
                             p_fct_reward = p_fct_reward,
                             p_fct_success = p_fct_success,
                             p_fct_broken = p_fct_broken,
                             p_mujoco_file = p_mujoco_file,
                             p_frame_skip = p_frame_skip,
                             p_state_mapping = p_state_mapping,
                             p_action_mapping = p_action_mapping,
                             p_camera_conf = p_camera_conf,
                             p_visualize = p_visualize,
                             p_logging = p_logging)

        OASystem.__init__(self,
                            p_id=p_id,
                            p_name=p_name,
                            p_range_max=p_range_max,
                            p_autorun=p_autorun,
                            p_class_shared=p_class_shared,
                            p_ada=p_ada,
                            p_mode=p_mode,
                            p_latency=p_latency,
                            p_t_step=p_t_step,
                            p_fct_strans=p_fct_strans,
                            p_fct_success=p_fct_success,
                            p_fct_broken=p_fct_broken,
                            p_wf=p_wf,
                            p_wf_success=p_wf_success,
                            p_wf_broken=p_wf_broken,
                            p_mujoco_file=p_mujoco_file,
                            p_frame_skip=p_frame_skip,
                            p_state_mapping=p_state_mapping,
                            p_action_mapping=p_action_mapping,
                            p_camera_conf=p_camera_conf,
                            p_visualize=p_visualize,
                            p_logging=p_logging,
                            **p_kwargs)


        self._workflows.append(self._wf_reward)
        self._fcts.append(self._fct_reward)


## -------------------------------------------------------------------------------------------------
    def compute_reward(self, p_state_old: State = None, p_state_new: State = None) -> Reward:
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
        if p_state_old is not None:
            state_old = p_state_old
        else:
            state_old = self._prev_state

        if state_old is None:
            return None

        if p_state_new is not None:
            state_new = p_state_new

        else:
            state_new = self._state

        if self._fct_reward is not None:
            self._last_reward = self._fct_reward.compute_reward(p_state_new=state_new, p_state_old=state_old)
        else:
            self._last_reward = OAFctReward.compute_reward(self, p_state_new=state_new, p_state_old=state_old)

        return self._last_reward


## -------------------------------------------------------------------------------------------------
    def get_workflow_reward(self):

        return self._wf_reward

