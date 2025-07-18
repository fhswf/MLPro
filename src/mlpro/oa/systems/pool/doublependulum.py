## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - The integrative middleware framework for standardized machine learning
## -- Package : mlpro.oa.systems.pool
## -- Module  : doublependulum.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2023-05-30  1.0.0     LSB      Creation
## -- 2023-06-07  1.0.1     LSB      Refactoring due to removal of DP at BF-ML-Pool level
## -- 2025-07-18  1.1.0     DA       Refactoring
## -------------------------------------------------------------------------------------------------

"""
Ver. 1.1.0 (2025-07-18)

This module provides the online adaptive extensions of the Double Pendulum System.

"""


from datetime import timedelta

from mlpro.bf import Log, Mode
from mlpro.bf.mt import *
from mlpro.bf.systems import FctSTrans, FctSuccess, FctBroken
from mlpro.bf.systems.pool.doublependulum import *

from mlpro.oa.streams import OAStreamWorkflow
from mlpro.oa.systems import *



# Export list for public API
__all__ = [ 'DoublePendulumOA4',
            'DoublePendulumOA7' ]




## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class DoublePendulumOA4(OASystem, DoublePendulumSystemS4):

    C_NAME = 'DoublePendulumOA4'

    def __init__(self,
                 p_id = None,
                 p_name: str = None,
                 p_range_max: int = Range.C_RANGE_NONE,
                 p_autorun: int = Task.C_AUTORUN_NONE,
                 p_class_shared: Shared = None,
                 p_ada: bool = True,
                 p_mode: int = Mode.C_MODE_SIM,
                 p_latency: timedelta = None,
                 p_t_step: timedelta = None,
                 p_max_torque: float = 20,
                 p_l1: float = 1.0,
                 p_l2: float = 1.0,
                 p_m1: float = 1.0,
                 p_m2: float = 1.0,
                 p_init_angles = DoublePendulumSystemRoot.C_ANGLES_RND,
                 p_g: float = 9.8,
                 p_fct_strans: FctSTrans = None,
                 p_fct_success: FctSuccess = None,
                 p_fct_broken: FctBroken = None,
                 p_wf: OAStreamWorkflow = None,
                 p_wf_success: OAStreamWorkflow = None,
                 p_wf_broken: OAStreamWorkflow = None,
                 p_mujoco_file = None,
                 p_frame_skip: int = None,
                 p_state_mapping = None,
                 p_action_mapping = None,
                 p_camera_conf: tuple = None,
                 p_history_length: int = 5,
                 p_random_range: list = None,
                 p_balancing_range: list = (-0.2,0.2),
                 p_swinging_outer_pole_range = (0.2,0.5),
                 p_break_swinging: bool = False,
                 p_visualize: bool = False,
                 p_logging: bool = Log.C_LOG_ALL,
                 **p_kwargs):

        self._max_torque = p_max_torque

        OASystem.__init__(self,
                             p_id = p_id,
                             p_name = p_name,
                             p_range_max = p_range_max,
                             p_autorun = p_autorun,
                             p_class_shared = p_class_shared,
                             p_ada = p_ada,
                             p_mode = p_mode,
                             p_latency = p_latency,
                             p_t_step = p_t_step,
                             p_fct_strans = p_fct_strans,
                             p_fct_success = p_fct_success,
                             p_fct_broken = p_fct_broken,
                             p_wf = p_wf,
                             p_wf_success = p_wf_success,
                             p_wf_broken = p_wf_broken,
                             p_mujoco_file = p_mujoco_file,
                             p_frame_skip = p_frame_skip,
                             p_state_mapping = p_state_mapping,
                             p_action_mapping = p_action_mapping,
                             p_camera_conf = p_camera_conf,
                             p_visualize = p_visualize,
                             p_logging = p_logging,
                             **p_kwargs)

        DoublePendulumSystemS4.__init__(   self,
                                     p_id = p_id,
                                     p_name = p_name,
                                     p_range_max = p_range_max,
                                     p_autorun = p_autorun,
                                     p_class_shared = p_class_shared,
                                     p_ada = p_ada,
                                     p_mode = p_mode,
                                     p_latency = p_latency,
                                     p_t_step = p_t_step,
                                     p_max_torque = p_max_torque,
                                     p_l1 = p_l1,
                                     p_l2 = p_l2,
                                     p_m1= p_m1,
                                     p_m2 = p_m2,
                                     p_init_angles = p_init_angles,
                                     p_g = p_g,
                                     p_fct_strans = p_fct_strans,
                                     p_fct_success = p_fct_success,
                                     p_fct_broken = p_fct_broken,
                                     p_mujoco_file = p_mujoco_file,
                                     p_frame_skip = p_frame_skip,
                                     p_state_mapping = p_state_mapping,
                                     p_action_mapping = p_action_mapping,
                                     p_camera_conf = p_camera_conf,
                                     p_history_length = p_history_length,
                                     p_visualize = p_visualize,
                                     p_random_range = p_random_range,
                                     p_balancing_range = p_balancing_range,
                                     p_swinging_outer_pole_range = p_swinging_outer_pole_range,
                                     p_break_swinging = p_break_swinging,
                                     p_logging = p_logging,
                                     **p_kwargs)







## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class DoublePendulumOA7(DoublePendulumSystemS7, DoublePendulumOA4):

    C_NAME = 'DoublePendulumOA7'



