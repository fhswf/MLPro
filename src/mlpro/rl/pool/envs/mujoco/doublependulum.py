## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - A Synoptic Framework for Standardized Machine Learning Tasks
## -- Package : mlpro.rl.pool.envs.mujoco
## -- Module  : doublependulum.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2022-09-19  0.0.0     MRD       Creation
## -------------------------------------------------------------------------------------------------


from mlpro.rl.models import *
from mlpro.wrappers.mujoco import WrEnvMujoco




## ---------------------------------------------------------------------------------------------------------------------
## ---------------------------------------------------------------------------------------------------------------------
class DoublePendulum(WrEnvMujoco):
    def __init__(self, p_frame_skip=1, p_logging=False):
        p_model_path = None
        super().__init__(p_model_path, p_frame_skip, p_logging)


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
                    p_unit='radian', p_unit_latex='\textdegrees', p_boundaries=[-180, 180]))
        state_space.add_dim(
            Dimension(p_name_long='theta 2', p_name_short='th2', p_description='Angle of pendulum 2', p_name_latex='',
                    p_unit='radian', p_unit_latex='\textdegrees', p_boundaries=[-180, 180]))
        state_space.add_dim(
            Dimension(p_name_long='omega 1', p_name_short='w1', p_description='Angular Velocity of Pendulum 1',
                    p_name_latex='', p_unit='radian/second', p_unit_latex='\textdegrees/s',p_boundaries=[-800, 800]))
        state_space.add_dim(
            Dimension(p_name_long='omega 2', p_name_short='w2', p_description='Angular Velocity of Pendulum 2',
                    p_name_latex='', p_unit='radian/second', p_unit_latex='\textdegrees/s', p_boundaries=[-950, 950]))
        

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
        self._step_simulation(action, self.frame_skip)
        joint_angle = self.data.qpos.flat[:]
        joint_velocity = self.data.qvel.flat[:]

        current_state = State(self._state_space)

        return current_state


## ------------------------------------------------------------------------------------------------------
    def _compute_reward(self, p_state_old: State, p_state_new: State) -> Reward:
        pass


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

        goal_state = self._target_state
        d = self.get_state_space().distance(p_state, goal_state)
        if d < self.C_THRSH_GOAL: return True
        return False


## ------------------------------------------------------------------------------------------------------
    def init_plot(self, p_figure=None):
        pass


## ------------------------------------------------------------------------------------------------------
    def update_plot(self):
        return super().update_plot()