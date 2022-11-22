## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - A Synoptic Framework for Standardized Machine Learning Tasks
## -- Package : mlpro.rl.pool.envs.mujoco
## -- Module  : doublependulum.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2022-09-19  0.0.0     MRD       Creation
## -------------------------------------------------------------------------------------------------


import numpy as np

from mlpro.rl.models import *
from mlpro.wrappers.mujoco import WrEnvMujoco




## ---------------------------------------------------------------------------------------------------------------------
## ---------------------------------------------------------------------------------------------------------------------
class DoublePendulum(WrEnvMujoco):
    def __init__(self, p_frame_skip=1, p_logging=False):
        p_model_path = None
        p_model_file = "doublependulum.xml"
        super().__init__(p_model_file, p_frame_skip=p_frame_skip, p_model_path=p_model_path, p_logging=p_logging)

        self._state = State(self._state_space)

        self.reset()


## ------------------------------------------------------------------------------------------------------
    def setup_spaces(self):
        """
        Method to setup the spaces for the Double Pendulum root environment. This method sets up four dimensional
        Euclidean space for the root DP environment.
        """

        state_space = ESpace()
        action_space = ESpace()

        state_space.add_dim(
            Dimension(p_name_long='theta 1', p_name_short='th1', p_description='Angle of Pendulum 1', p_name_latex='',
                    p_unit='radian', p_unit_latex='\textdegrees'))
        state_space.add_dim(
            Dimension(p_name_long='theta 2', p_name_short='th2', p_description='Angle of pendulum 2', p_name_latex='',
                    p_unit='radian', p_unit_latex='\textdegrees'))
        state_space.add_dim(
            Dimension(p_name_long='omega 1', p_name_short='w1', p_description='Angular Velocity of Pendulum 1',
                    p_name_latex='', p_unit='radian/second', p_unit_latex='\textdegrees/s'))
        state_space.add_dim(
            Dimension(p_name_long='omega 2', p_name_short='w2', p_description='Angular Velocity of Pendulum 2',
                    p_name_latex='', p_unit='radian/second', p_unit_latex='\textdegrees/s'))

        action_space.add_dim(
            Dimension(p_name_long='torque', p_name_short='tau', p_description='Torque Joint 1',
                    p_name_latex='', p_unit='radian/second', p_unit_latex='\textdegrees/s', p_boundaries=[-1, 1])
                    )
        

        return state_space, action_space


## ------------------------------------------------------------------------------------------------------
    def _reset(self, p_seed=None) -> None:
        """
        This method is used to reset the environment. The environment is reset to the initial position set during
        the initialization of the environment.

        Parameters
        ----------
        p_seed : int, optional
            The default is None.

        """
        
        self._reset_simulation()

        ob = self.reset_model()
        self.render()

        self._state.set_values(ob)

## ------------------------------------------------------------------------------------------------------
    def reset_model(self):
        qpos = self.init_qpos + np.random.uniform(
            size=self.model.nq, low=-0.01, high=0.01
        )
        qvel = self.init_qvel + np.random.uniform(
            size=self.model.nv, low=-0.01, high=0.01
        )
        self.set_state(qpos, qvel)
        return self._get_obs()

## ------------------------------------------------------------------------------------------------------
    def _get_obs(self):
        return np.concatenate([self.data.qpos, self.data.qvel]).ravel()


## ------------------------------------------------------------------------------------------------------
    def _simulate_reaction(self, p_state:State, p_action:Action):
        """
        This method is used to calculate the next states of the system after a set of actions.

        Parameters
        ----------
        p_state : State
            current State.
            p_action : Action
                current Action.

        Returns
        -------
            _state : State
                Current states after the simulation of latest action on the environment.

        """
        action = p_action.get_sorted_values()
        self._step_simulation(action)
        joint_angle = self.data.qpos.flat[:]
        joint_velocity = self.data.qvel.flat[:]

        current_state = State(self._state_space)
        current_state.set_values([*joint_angle, *joint_velocity])

        return current_state


## ------------------------------------------------------------------------------------------------------
    def _compute_reward(self, p_state_old: State, p_state_new: State) -> Reward:
        current_reward = Reward()
        current_reward.set_overall_reward(1)
        return current_reward


## ------------------------------------------------------------------------------------------------------
    def _compute_broken(self, p_state: State) -> bool:
        """
        Custom method to compute broken state. In this case always returns false as the environment doesn't break
        """

        return False


## ------------------------------------------------------------------------------------------------------
    def _compute_success(self, p_state:State):
        """
        Custom method to return the success state of the environment based on the distance between current state,
        goal state and the goal threshold parameter

        Parameters
        ----------
        p_state:State
            current state of the environment

        Returns
        -------
        bool
            True if the distance between current state and goal state is less than the goal threshold else false
        """

        return False


## ------------------------------------------------------------------------------------------------------
    def init_plot(self, p_figure=None):
        pass


## ------------------------------------------------------------------------------------------------------
    def update_plot(self):
        return super().update_plot()