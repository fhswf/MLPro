## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - A Synoptic Framework for Standardized Machine Learning Tasks
## -- Package : mlpro.rl.pool.envs
## -- Module  : cartpole.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2023-02-23  0.0.0     MRD       Creation
## -- 2023-02-23  1.0.0     MRD       Released first version
## -------------------------------------------------------------------------------------------------

"""
Ver. 1.0.0 (2023-02-23)

This module provides a CartPole environment with MuJoCo Simulation. 
"""


import os
import mlpro
from mlpro.rl.models_env import Environment
from mlpro.rl.models_agents import Reward, State
from mlpro.bf.ops import Mode
from mlpro.bf.various import Log
from mlpro.bf.math import Dimension





## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class CartpoleMujoco(Environment):

    C_NAME = "CartpoleMujoco"
    C_REWARD_TYPE = Reward.C_TYPE_OVERALL
    C_CYCLE_LIMIT = 400

## -------------------------------------------------------------------------------------------------
    def __init__(
        self,
        p_visualize: bool = False,
        p_logging=Log.C_LOG_ALL,
    ):
        model_file = os.path.join(os.path.dirname(mlpro.__file__), "bf/systems/pool/mujoco", "cartpole.xml")
        super().__init__(
            p_mode=Mode.C_MODE_SIM,
            p_mujoco_file=model_file,
            p_frame_skip=1,
            p_state_mapping=None,
            p_action_mapping=None,
            p_camera_conf=(None, None, None),
            p_visualize=p_visualize,
            p_logging=p_logging,
        )

        self._state = State(self._state_space)
        self._max_action = self._action_space.get_dim_by_name("force").get_boundaries()[-1]
        self._change_action_space()
        self.reset()


## -------------------------------------------------------------------------------------------------
    def _change_action_space(self):
        pass


## -------------------------------------------------------------------------------------------------
    def _compute_reward(self, p_state_old: State = None, p_state_new: State = None) -> Reward:
        reward = Reward(self.C_REWARD_TYPE)
        reward.set_overall_reward(1)

        return reward


## -------------------------------------------------------------------------------------------------
    def _compute_broken(self, p_state: State) -> bool:
        state_value = p_state.get_values()

        slide_pos_joint_thresh = 3
        hinge_pos_joint_thresh = 0.3

        slide_pos_joint = state_value[0]
        hinge_pos_joint = state_value[3]

        terminated = bool(
            slide_pos_joint < -slide_pos_joint_thresh
            or slide_pos_joint > slide_pos_joint_thresh
            or hinge_pos_joint < -hinge_pos_joint_thresh
            or hinge_pos_joint > hinge_pos_joint_thresh
        )

        self._state.set_terminal(terminated)
        return terminated


## -------------------------------------------------------------------------------------------------
    def _reset(self, p_seed=None) -> None:
        pass





## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class CartpoleMujocoDiscrete(CartpoleMujoco):

    C_NAME = "CartpoleMujocoDiscrete"

## -------------------------------------------------------------------------------------------------
    def __init__(self, p_visualize: bool = False, p_logging=Log.C_LOG_ALL):
        super().__init__(p_visualize, p_logging)


## -------------------------------------------------------------------------------------------------
    def _change_action_space(self):
        self._action_space.get_dim_by_name("force")._base_set = Dimension.C_BASE_SET_Z
        self._action_space.get_dim_by_name("force").set_boundaries([0, 1])


## -------------------------------------------------------------------------------------------------
    def _action_to_mujoco(self, p_mlpro_action):
        action = self._max_action if p_mlpro_action == 1 else -self._max_action
        return action





## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class CartpoleMujocoContinuous(CartpoleMujoco):

    C_NAME = "CartpoleMujocoContinuous"
