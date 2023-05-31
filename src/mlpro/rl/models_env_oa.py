## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - A Synoptic Framework for Standardized Machine Learning Tasks
## -- Package : mlpro.rl
## -- Module  : models_adaptive_environment.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- yyyy-mm-dd  0.0.0     LSB      Creation
## -------------------------------------------------------------------------------------------------

"""
Ver. 0.0.0 (yyyy-mm-dd)

This module provides model classes for adaptive environments
"""


from mlpro.oa.streams import *
from mlpro.oa.systems import *
from mlpro.rl.models import *





## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class OAFctReward(FctReward, Model):
    """
    Online adaptive function for reward computation.
    Parameters
    ----------
    p_name
    p_range_max
    p_class_shared
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
                 p_wf_reward: OAWorkflow = None,
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
            self._wf_reward = OAWorkflow(p_visualize=p_visualize,
                                         p_ada=p_ada,
                                         p_logging=p_logging)
        else:
            self._wf_reward = p_wf_reward


        self._state_obj_old:State = None
        self._state_obj_new:State = None
        self._setup_wf_reward = False


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


        self.log(Log.C_LOG_TYPE_I, 'Start Computing the Reward....')
        #
        #
        # # 1. check if the user has already created a workflow and added to tasks
        # if self._processing_wf is None:
        #     # Create a shared object if not provided
        #     if self._shared is None:
        #         self._shared = StreamShared()
        #
        #     # Create an OA workflow
        #     self._processing_wf = OAWorkflow(p_name = 'Reward Wf', p_class_shared=self._shared)
        #
        #
        # # 2. Create a reward task
        # if self._reward_task is None:
        #
        #     # Create a pseudo reward task
        #     self._reward_task = OATask(p_name='Compute Reward',
        #                                p_visualize = self._visualize,
        #                                p_range_max=self.get_range(),
        #                                p_duplicate_data = True)
        #
        #     # Assign the task method to custom implementation
        #     self._reward_task._run = self._run
        #
        #     # Add the task to workflow
        #     self._processing_wf.add_task(self._reward_task)
        #
        #
        #
        # # 4. Creating task level attributes for states
        # try:
        #     self._reward_task._state
        # except AttributeError:
        #     self._reward_task._state = p_state.copy()
        # try:
        #     self._reward_task._state_new
        # except AttributeError:
        #     self._reward_task._state_new = p_state_new.copy()
        #
        #
        #
        # # 5. Creating new and old instances
        # # creating old instance object if this is the first run
        # if self._instance_new == None:
        #     self._instance_old = Instance(p_state)
        #
        # # assigning the previous new instance to old instance
        # else:
        #     self._instance_old = self._instance_new.copy()
        #
        # # creating new instance with new state
        # self._instance_new = Instance(p_state_new)
        if p_state_old is not None:
            self._state_obj_old = p_state_old.copy()
        self._state_obj_new = p_state_new.copy()
        self.log(Log.C_LOG_TYPE_I, 'Assessing Broken...')

        if not self._setup_wf_reward:
            self._setup_wf_reward = self._setup_oafct_reward()

        # 6. Run the workflow
        self._wf_reward.run(p_inst_new=[self._state_obj_new, self._state_obj_old])

        # 7. Return the results
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
    def add_task_success(self, p_task: StreamTask, p_pred_tasks: list = None):
        """
        Adds a task to the workflow.
        Parameters
        ----------
        p_task: OATask, StreamTask
            The task to be added to the workflow

        p_pred_task: list[Task]
            Name of the predecessor tasks for the task to be added

        """

        self._wf_reward.add_task(p_task=p_task, p_pred_tasks=p_pred_tasks)

## -------------------------------------------------------------------------------------------------
    def _run_wf_reward(self, p_inst_new, p_inst_del):
        """
        Runs the reward computation workflow of the system.

        Parameters
        ----------
        p_inst_new: list[State]
            List of new instances to be processed by the workflow.

        p_inst_del: list[State]
            List of old instances to be processed by the workflow.

        """

        if self._afct_reward is not None:
            self._wf_reward.get_so().add_result(self.get_id(), AFctReward.compute_reward(self,
                                                                              p_state_new=p_inst_new[0],
                                                                              p_state_old=p_inst_new[1]))
        else:
            self._wf_reward.get_so().add_result(self.get_id(), FctReward.compute_reward(self,
                                                                             p_state_new=p_inst_new[0],
                                                                             p_state_old=p_inst_new[1]))

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
            p_pred_tasks = self._wf_reward._tasks[-1]

        self._wf_reward.add_task(p_task=PseudoTask(p_wrap_method=self._run_wf_reward),
                                  p_pred_tasks=p_pred_tasks)
        return True






## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class OAEnvironment(OAFctReward, OASystem, Environment):
    """

    Parameters
    ----------
    p_mode
    p_latency
    p_ada
    p_fct_strans
    p_fct_reward
    p_fct_success
    p_fct_broken
    p_visualize
    p_logging
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
                 p_wf : OAWorkflow = None,
                 p_wf_success : OAWorkflow = None,
                 p_wf_broken : OAWorkflow = None,
                 p_wf_reward : OAWorkflow = None,
                 p_mujoco_file = None,
                 p_frame_skip: int = 1,
                 p_state_mapping = None,
                 p_action_mapping = None,
                 p_camera_conf: tuple = (None, None, None),
                 p_visualize: bool = False,
                 p_logging: bool = Log.C_LOG_ALL,
                 **p_kwargs):



        OAFctReward.__init__(self, p_wf_reward=p_wf_reward)

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


# ## -------------------------------------------------------------------------------------------------
#     def _set_adapted(self, p_adapted:bool):
#         """
#
#         Parameters
#         ----------
#         p_adapted
#
#         Returns
#         -------
#
#         """
#         pass


# ## -------------------------------------------------------------------------------------------------
#     def _adapt(self, **p_kwargs):
#
#           This is exactly the same method as that of OASystem, thus I dont need it here,
#           since I have already added the functions and workflows to the list
#
#         adapted = False
#
#         for workflow in self._workflows:
#             try:
#                 adapted = workflow.adapt(**p_kwargs) or adapted
#             except:
#                 pass
#
#         for fct in self._fcts:
#             try:
#                 adapted = fct.adapt(**p_kwargs) or adapted
#             except:
#                 pass
#
#
#         return adapted
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
