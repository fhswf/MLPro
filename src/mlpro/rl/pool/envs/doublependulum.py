## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - The integrative middleware framework for standardized machine learning
## -- Package : mlpro.rl.envs
## -- Module  : doublependulum.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2022-01-25  0.0.0     WB       Creation
## -- 2022-01-26  0.9.0     WB       Initial trial of the environment
## -- 2022-01-27  0.9.1     WB       Trial without animation 
## -- 2022-01-28  0.9.2     WB       Fix the  update_plot method
## -- 2022-01-31  0.9.3     WB       Taking account of the new state in _compute_reward method 
## -- 2022-01-31  0.9.4     WB       Add Circular arrow to the plot 
## -- 2022-02-02  1.0.0     WB       Release of first version
## -- 2022-02-02  1.0.1     MRD      Cleaning the code
## -- 2022-02-10  1.0.2     WB       Introduce transparency in arrow depending on applied torque
## -- 2022-02-10  1.0.3     WB       Set init_angles as presets for starting angles
## -- 2022-02-10  1.0.4     WB       Normalize angle in reward calculation
## -- 2022-02-10  1.0.5     WB       Fix arrow head 
## -- 2022-02-14  1.0.6     WB       Update _compute_reward method
## -- 2022-02-17  1.0.7     WB       Taking into account the outer pole in reward calculation
## -- 2022-02-21  1.0.8     WB       Edit the formulation the of _compute_reward method
## -- 2022-03-02  1.0.9     WB       Include Torque and Change of state in _compute_reward method
## -- 2022-04-08  1.1.0     SY       Refactoring due to auto generated ID in class Dimension
## -- 2022-04-19  1.1.1     YI       Editing the State Space and Normalization of State Values
## -- 2022-05-10  1.1.2     YI       Debugging
## -- 2022-05-14  1.1.3     YI       Scalling manually
## -- 2022-05-16  1.1.4     SY       Code cleaning
## -- 2022-05-19  1.1.5     YI       Editing the reward function
## -- 2022-05-28  1.1.6     YI       Editing the reward, normalization, and derivs function
## -- 2022-05-30  1.1.7     SY       Enhance data normalization method, reset method, and code cleaning
## -- 2022-06-21  1.1.8     SY       Code cleaning
## -- 2022-07-10  1.1.9     YI       Changing the units from radians to degrees
## -- 2022-07-20  1.2.0     SY       Updating _simulate_reaction, debugging _data_normalization
## -- 2022-07-28  1.3.0     LSB      Updating:
##                                      - Numericals for torque and acceleration
##                                      - Updating visualisation
## -- 2022-07-28  1.3.1     LSB      Returning new state object at simulate reaction method
## -- 2022-08-01  1.3.2     LSB      Coverting radians to degrees only in the state space
## -- 2022-08-05  1.3.3     LSB      Limiting the th1 and th2 within 180 to -180 degrees
## -- 2022-08-05  1.3.4     SY       Refactoring
## -- 2022-08-05  1.3.5     YI       Updating the bondaries of the environment's states
## -- 2022-08-14  1.3.6     LSB      - Minor change in the max torque value, step size in integration
##                                   - Inverted angles with 0 degrees at top
## -- 2022-08-05  1.3.7     SY       Minor changing: Boundaries of the pendulums' angle
## -- 2022-08-19  1.4.7     LSB      Classic Variant inherited from the root DP
## -- 2022-08-20  1.4.8     LSB      New attribute target state
## -- 2022-08-29  2.0.0     LSB      - New varients to be used
##                                   - Default reward strategy and success strategy and bug fixes
## -- 2022-09-02  2.0.1     LSB      Refactoring, code cleaning
## -- 2022-09-05  2.0.2     LSB      Refactoring
## -- 2022-09-06  2.0.3     LSB/DA   Refactoring
## -- 2022-09-09  2.0.4     SY       Updating reward function and compute success function
## -- 2022-09-09  2.0.5     LSB      Updating the boundaries
## -- 2022-10-08  2.0.6     LSB      Bug fix
## -- 2022-11-09  2.1.0     DA       Refactorung due to changes on the plot systematics
## -- 2022-11-11  2.1.1     LSB      Bug fix for random seed dependent reproducibility
## -- 2022-11-17  2.2.0     LSB      New plot systematics
## -- 2022-11-18  2.2.1     DA       Method DoublePendulumRoot._init_figure(): title
## -- 2022-11-18  2.2.2     LSB      reward strategies, broken computation, balancing and swinging
## -- 2022-11-20  2.2.3     LSB      Reward Window, reward trend, reward strategies, balancing
##                                   and swinging zones
## -- 2022-12-07  2.2.4     LSB      - updated the reward strategy
##                                   - removed constructor of S4 class
## -- 2023-02-02  2.2.5     DA       Removed method DoublePendulumRoot._init_figure()
## -- 2023-03-01  2.2.6     LSB      Weight components for state space
## -- 2023-03-03  2.2.7     LSB      Bug Fix
## -- 2023-03-05  2.3.0     LSB      Shifted the environment into a system in bf systems pool
## -- 2023-03-08  2.3.1     LSB      Refactoring for visualization
## -- 2023-03-09  2.3.2     LSB      Minor Bug Fix
## -- 2023-05-30  3.0.0     LSB      Adaptive Extensions for Double Pendulum:
##                                       - DoublePendulumA4
##                                       - DoublePendulumA7
## -- 2025-07-17  3.1.0     DA       Refactoring
## -------------------------------------------------------------------------------------------------

"""
Ver. 3.1.0 (2025-07-17) 

The Double Pendulum environment is an implementation of a classic control problem of Double Pendulum system. The
dynamics of the system are based on the `Double Pendulum <https://matplotlib.org/stable/gallery/animation/double_pendulum.html>`_  implementation by
`Matplotlib <https://matplotlib.org/>`_. The double pendulum is a system of two poles, with the inner pole
connected to a fixed point at one end and to outer pole at other end. The native implementation of Double
Pendulum consists of an input motor providing the torque in either directions to actuate the system.
"""


from datetime import timedelta  

import numpy as np

from mlpro.bf import *
from mlpro.bf.mt import *
from mlpro.bf.plot import Figure
from mlpro.bf.systems import *

from mlpro.rl import *
from mlpro.bf.systems.pool.doublependulum import *
from mlpro.rl.models_env_oa import *

from mlpro.oa.streams import OAStreamWorkflow
import mlpro.oa.systems.pool.doublependulum as oadp



# Export list for public API
__all__ = [ 'DoublePendulumRoot', 
            'DoublePendulumS4', 
            'DoublePendulumS7',
            'DoublePendulumOA4',
            'DoublePendulumOA7' ]




## ---------------------------------------------------------------------------------------------------------------------
## ---------------------------------------------------------------------------------------------------------------------
class DoublePendulumRoot (DoublePendulumSystemRoot, Environment):
    """
    This is the root double pendulum environment class inherited from Environment class with four dimensional state
    space and underlying implementation of the Double Pendulum dynamics, default reward strategy.

    Parameters
    ----------
    p_mode 
        Mode of environment. Possible values are Mode.C_MODE_SIM(default) or Mode.C_MODE_REAL.
    p_latency : timedelta
        Optional latency of environment. If not provided, the internal value of constant C_LATENCY is used by default.
    p_max_torque : float, optional
        Maximum torque applied to pendulum. The default is 20.
    p_l1 : float, optional
        Length of pendulum 1 in m. The default is 0.5
    p_l2 : float, optional
        Length of pendulum 2 in m. The default is 0.25
    p_m1 : float, optional
        Mass of pendulum 1 in kg. The default is 0.5
    p_m2 : float, optional
        Mass of pendulum 2 in kg. The default is 0.25
    p_g : float, optional
        Gravitational acceleration. The default is 9.8
    p_init_angles: str, optional
        C_ANGLES_UP starts the pendulum in an upright position
        C_ANGLES_DOWN starts the pendulum in a downward position
        C_ANGLES_RND starts the pendulum from a random position.
    p_history_length : int, optional
        Historical trajectory points to display. The default is 5.
    p_fct_strans: FctSTrans, optional
        The custom State transition function.
    p_fct_success: FctSuccess, optional
        The custom Success Function.
    p_fct_broken: FctBroken, optional
        The custom Broken Function.
    p_fct_reward:FctReward, optional
        The custom Reward Function.
    p_mujoco_file: optional
        The corresponding mujoco file
    p_frame_skip: optional
        Number of frames to be skipped for visualization.
    p_state_mapping: optional
        State mapping configurations.
    p_action_mapping: optional
        Action mapping configurations.
    p_camera_conf: optional
        Camera configurations for mujoco specific visualization.
    p_visualize : bool
        Boolean switch for visualisation. Default = False.
    p_plot_level : int
        Types and number of plots to be plotted. Default = ALL
        C_PLOT_DEPTH_ENV only plots the environment
        C_PLOT_DEPTH_REWARD only plots the reward
        C_PLOT_ALL plots both reward and the environment
    p_rst_balancing
        Reward strategy to be used for the balancing region of the environment
    p_rst_swinging
        Reward strategy to be used for the swinging region of the environment
    p_rst_swinging_outer_pole
        Reward Strategy to be used for swinging up outer pole
    p_reward_weights:list
        List of weights to be added to the dimensions of the state space for reward computation
    p_reward_trend:bool
        Boolean value stating whether to plot reward trend
    p_reward_window:int
        The number of latest rewards to be shown in the plot. Default is 0
    p_random_range:list
        The boundaries for state space for initialization of environment randomly
    p_balancing range:list
        The boundaries for state space of environment in balancing region
    p_swinging_outer_pole_range
        The boundaries for state space of environment in swinging of outer pole region
    p_break_swinging:bool
        Boolean value stating whether the environment shall be broken outside the balancing region
    p_logging
        Log level (see constants of class mlpro.bf.various.Log). Default = Log.C_LOG_WE.
    """


    C_REWARD_TYPE       = Reward.C_TYPE_OVERALL


    C_PLOT_DEPTH_ENV    = 0
    C_PLOT_DEPTH_REWARD = 1
    C_PLOT_DEPTH_ALL    = 2
    C_VALID_DEPTH       = [C_PLOT_DEPTH_ENV, C_PLOT_DEPTH_REWARD, C_PLOT_DEPTH_ALL]

    C_RST_BALANCING_001 = 'rst_001'
    C_RST_BALANCING_002 = 'rst_002'
    C_RST_SWINGING_001  = 'rst_003'
    C_RST_SWINGING_OUTER_POLE_001 = 'rst_004'
    C_VALID_RST_BALANCING = ['rst_001','rst_002']
    C_VALID_RST_SWINGING = ['rst_003']
    C_VALID_RST_SWINGING_OUTER_POLE = ['rst_004']

## -------------------------------------------------------------------------------------------------
    def __init__ ( self,
                   p_id=None,
                   p_name=None,
                   p_buffer_size=0,
                   p_range_max=Range.C_RANGE_NONE,
                   p_autorun=Task.C_AUTORUN_NONE,
                   p_class_shared=None,
                   p_mode = Mode.C_MODE_SIM,
                   p_latency = None,
                   p_max_torque=20,
                   p_l1=1.0,
                   p_l2=1.0,
                   p_m1=1.0,
                   p_m2=1.0,
                   p_g=9.8,
                   p_init_angles=DoublePendulumSystemRoot.C_ANGLES_RND,
                   p_history_length=5,
                   p_fct_strans:FctSTrans=None,
                   p_fct_success:FctSuccess=None,
                   p_fct_broken:FctBroken=None,
                   p_fct_reward:FctReward=None,
                   p_mujoco_file=None,
                   p_frame_skip=None,
                   p_state_mapping=None,
                   p_action_mapping=None,
                   p_camera_conf=None,
                   p_visualize:bool=False,
                   p_plot_level:int=2,
                   p_rst_balancing = C_RST_BALANCING_002,
                   p_rst_swinging = C_RST_SWINGING_001,
                   p_rst_swinging_outer_pole = C_RST_SWINGING_OUTER_POLE_001,
                   p_reward_weights: list = None,
                   p_reward_trend: bool = False,
                   p_reward_window:int = 0,
                   p_random_range:list = None,
                   p_balancing_range:list = (-0.2,0.2),
                   p_swinging_outer_pole_range = (0.2,0.5),
                   p_break_swinging:bool = False,
                   p_logging=Log.C_LOG_ALL ):


        DoublePendulumSystemRoot.__init__(self,p_id = p_id,
                         p_name=p_name,
                         p_buffer_size = p_buffer_size,
                         p_range_max = p_range_max,
                         p_autorun = p_autorun,
                         p_class_shared = p_class_shared,
                         p_mode = p_mode,
                         p_latency = p_latency,
                         p_max_torque=p_max_torque,
                         p_l1=p_l1,
                         p_l2=p_l2,
                         p_m1=p_m1,
                         p_m2=p_m2,
                         p_init_angles=p_init_angles,
                         p_g=p_g,
                         p_history_length=p_history_length,
                         p_random_range = p_random_range,
                         p_balancing_range = p_balancing_range,
                         p_swinging_outer_pole_range = p_swinging_outer_pole_range,
                         p_break_swinging = p_break_swinging,
                         p_visualize = p_visualize,
                         p_logging = p_logging)


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


        self._t_step = self._t_step = self.get_latency().seconds + self.get_latency().microseconds / 1000000
        self._state = State(self._state_space)
        self._target_state = State(self._state_space)
        self._target_state.set_values(np.zeros(self._state_space.get_num_dim()))

        self._rst_balancing = p_rst_balancing
        self._rst_swinging = p_rst_swinging
        self._rst_swinging_outer_pole = p_rst_swinging_outer_pole
        self._rst_methods = {DoublePendulumRoot.C_RST_BALANCING_001:self.compute_reward_001,
                             DoublePendulumRoot.C_RST_BALANCING_002:self.compute_reward_002,
                             DoublePendulumRoot.C_RST_SWINGING_001:self.compute_reward_003,
                             DoublePendulumRoot.C_RST_SWINGING_OUTER_POLE_001:self.compute_reward_004}
        self._plot_level = p_plot_level

        if self._plot_level in [self.C_PLOT_DEPTH_REWARD, self.C_PLOT_DEPTH_ALL]:
            self._reward_history = []
            self._reward_trend = p_reward_trend
            self._reward_window = p_reward_window

        if p_reward_weights is None:
            self._reward_weights = [1 for i in range(len(self.get_state_space().get_dims()))]
        self.reset()


## ------------------------------------------------------------------------------------------------------
    def _compute_reward(self, p_state_old: State = None, p_state_new: State = None) -> Reward:
        """
        This method calculates the reward for C_TYPE_OVERALL reward type. The current reward is based on the
        worst possible distance between two states and the best possible minimum distance between current and
        goal state.

        Parameters
        ----------
        p_state_old : State
            Previous state.
        p_state_new : State
            New state.

        Returns
        -------
        current_reward : Reward
            current calculated Reward values.
        """

        state_new = p_state_new.get_values().copy()
        p_state_new_normalized = self._normalize(state_new)
        norm_state_new = State(self.get_state_space())
        norm_state_new.set_values(p_state_new_normalized)

        state_old = p_state_old.get_values().copy()
        p_state_old_normalized = self._normalize(state_old)
        norm_state_old = State(self.get_state_space())
        norm_state_old.set_values(p_state_old_normalized)

        if (abs(p_state_new_normalized[0]) <= 0.2 and abs(p_state_new_normalized[2] <= 0.2)):
            if self._rst_balancing in self.C_VALID_RST_BALANCING:
                current_reward = self._rst_methods[self._rst_balancing](norm_state_old, norm_state_new)
            else:
                raise AttributeError('Reward strategy does not exist.')

        elif (0.5 >= abs(p_state_new_normalized[0]) > 0.2):
            if self._rst_swinging_outer_pole in self.C_VALID_RST_SWINGING_OUTER_POLE:
                current_reward = self._rst_methods[self._rst_swinging_outer_pole](norm_state_old, norm_state_new)
            else:
                raise AttributeError('Reward strategy does not exist.')

        else:
            if self._rst_swinging in self.C_VALID_RST_SWINGING:
                current_reward = self._rst_methods[self._rst_swinging](norm_state_old, norm_state_new)
            else:
                raise AttributeError('Reward strategy does not exist.')

        if self._plot_level in [self.C_PLOT_DEPTH_REWARD, self.C_PLOT_DEPTH_ALL]:
            if self._reward_window != 0 and len(self._reward_history) >= self._reward_window:
                self._reward_history.pop(0)
            self._reward_history.append(current_reward.overall_reward)
        return current_reward


## ------------------------------------------------------------------------------------------------------
    def compute_reward_001(self, p_state_old:State=None, p_state_new:State=None):
        """
        Reward strategy with only new normalized state

        Parameters
        ----------
        p_state_old : State
            Normalized old state.
        p_state_new : State
            Normalized new state.

        Returns
        -------
        current_reward : Reward
            current calculated Reward values.
        """
        current_reward = Reward()

        goal_state = self._target_state

        max_values = []
        min_values = []
        for i in self._state_space.get_dim_ids():
            boundaries = self._state_space.get_dim(i).get_boundaries()
            max_values.append(boundaries[1])
            min_values.append(boundaries[0])

        max_values = self._normalize(max_values)
        max_state = State(self.get_state_space())
        max_state.set_values(max_values)
        min_values = self._normalize(min_values)
        min_state = State(self.get_state_space())
        min_state.set_values(min_values)


        d = self.get_state_space().distance(p_state_new, goal_state)


        if d <= self.C_THRSH_GOAL:
            current_reward.set_overall_reward(1)
        else:
            current_reward.set_overall_reward(-d)

        return current_reward


## ------------------------------------------------------------------------------------------------------
    def compute_reward_002(self, p_state_old:State=None, p_state_new:State=None):
        """
        Reward strategy with both new and old normalized state based on euclidean distance from the goal state.
        Designed the balancing zone.

        Parameters
        ----------
        p_state_old : State
            Normalized old state.
        p_state_new : State
            Normalized new state.

        Returns
        -------
        current_reward : Reward
            current calculated Reward values.
        """
        current_reward = Reward()

        goal_state = self._target_state

        d_old = abs(self.get_state_space().distance(goal_state, p_state_old))
        d_new = abs(self.get_state_space().distance(goal_state, p_state_new))
        d = d_old - d_new

        current_reward.set_overall_reward(d)

        return current_reward


## ------------------------------------------------------------------------------------------------------
    def compute_reward_003(self, p_state_old: State = None, p_state_new: State = None):
        """
        Reward strategy with both new and old normalized state based on euclidean distance from the goal state,
        designed for the swinging of outer pole. Both angles and velocity and acceleration of the outer pole are
        considered for the reward computation.

        Parameters
        ----------
        p_state_old : State
            Normalized old state.
        p_state_new : State
            Normalized new state.

        Returns
        -------
        current_reward : Reward
            current calculated Reward values.
        """
        current_reward = Reward()
        state_new = p_state_new.get_values().copy()
        state_new[1] = 0
        state_new[4] = 0
        norm_state_new = State(self.get_state_space())
        norm_state_new.set_values(state_new)

        state_old = p_state_old.get_values().copy()
        state_old[1] = 0
        state_old[4] = 0
        norm_state_old = State(self.get_state_space())
        norm_state_old.set_values(state_old)
        goal_state = self._target_state

        d_old = abs(self.get_state_space().distance(goal_state, norm_state_old))
        d_new = abs(self.get_state_space().distance(goal_state, norm_state_new))
        d = d_old - d_new

        current_reward.set_overall_reward(d)

        return current_reward

## ------------------------------------------------------------------------------------------------------
    def compute_reward_004(self, p_state_old: State = None, p_state_new: State = None):
        """
        Reward strategy with both new and old normalized state based on euclidean distance from the goal state,
        designed for the swinging up region. Both angles and velocity and acceleration of the outer pole are
        considered for the reward computation.

        The reward strategy is as follows:

        reward = (|old_theta1n| - |new_theta1n|) + (|new_omega1n + new_alpha1n| - |old_omega1n + old_alpha1n|)

        Parameters
        ----------
        p_state_old : State
            Normalized old state.
        p_state_new : State
            Normalized new state.

        Returns
        -------
        current_reward : Reward
            current calculated Reward values.
        """
        current_reward = Reward()
        state_new = p_state_new.get_values().copy()
        p_state_new_normalized = self._normalize(state_new)


        state_old = p_state_old.get_values().copy()
        p_state_old_normalized = self._normalize(state_old)


        term_1 = abs(p_state_old_normalized[0] - p_state_new_normalized[0])

        try:
            term_2 = (abs(p_state_new_normalized[1]+p_state_new_normalized[4])
                        - abs(p_state_old_normalized[1]+p_state_old_normalized[4]))

        except:
            term_2 = (abs(p_state_new_normalized[1]) - abs(p_state_old_normalized[1]))

        current_reward.set_overall_reward(term_1+term_2)
        return current_reward


## ------------------------------------------------------------------------------------------------------
    def _init_plot_2d(self, p_figure: Figure, p_settings: PlotSettings):

        """
        Custom method to initialize a 2D plot. If attribute p_settings.axes is not None the
        initialization shall be done there. Otherwise a new MatPlotLib Axes object shall be
        created in the given figure and stored in p_settings.axes.

        Parameters
        ----------
        p_figure : Matplotlib.figure.Figure
            Matplotlib figure object to host the subplot(s).
        p_settings : PlotSettings
            Object with further plot settings.
        """

        p_settings.axes = []

        # Creates a grid space in the figure to use for subplot location
        if self._plot_level in [DoublePendulumRoot.C_PLOT_DEPTH_ENV, DoublePendulumRoot.C_PLOT_DEPTH_REWARD]:
            grid = p_figure.add_gridspec(1, 1)
        elif self._plot_level == DoublePendulumRoot.C_PLOT_DEPTH_ALL:
            if self._reward_trend:
                grid = p_figure.add_gridspec(1, 3)
                p_figure.set_size_inches(17, 5)
            else:
                grid = p_figure.add_gridspec(1,2)
                p_figure.set_size_inches(11, 5)

        if self._plot_level in [DoublePendulumRoot.C_PLOT_DEPTH_ENV, DoublePendulumRoot.C_PLOT_DEPTH_ALL]:
            DoublePendulumSystemRoot._init_plot_2d(self, p_figure = p_figure, p_settings = p_settings)
            p_settings.axes[0].set_subplotspec(grid[0])

        if self._plot_level in [DoublePendulumRoot.C_PLOT_DEPTH_REWARD,
                                DoublePendulumRoot.C_PLOT_DEPTH_ALL]:
            if self._reward_trend:
                if self._plot_level == DoublePendulumRoot.C_PLOT_DEPTH_ALL:
                    p_settings.axes.append(p_figure.add_subplot(grid[1:3]))
                else:
                    p_settings.axes.append(p_figure.add_subplot(111))
                self._plot_reward_trend, = p_settings.axes[-1].plot(range(len(self._reward_history)),
                    self._reward_history, 'r--', lw = 2)
            else:
                if self._plot_level == DoublePendulumRoot.C_PLOT_DEPTH_ALL:
                    p_settings.axes.append(p_figure.add_subplot(grid[1:3]))
                else:
                    p_settings.axes.append(p_figure.add_subplot(111))


            p_settings.axes[-1].set_title('Reward - '+ self.C_NAME)
            p_settings.axes[-1].autoscale_view()
            p_settings.axes[-1].grid()
            p_settings.axes[-1].set_xlabel('Cycle ID')
            p_settings.axes[-1].set_ylabel('Reward')
            self._reward_plot,  = p_settings.axes[-1].plot(range(len(self._reward_history)), self._reward_history,'b', lw = 1)


## ------------------------------------------------------------------------------------------------------
    def _update_plot_2d(self, p_settings: PlotSettings, **p_kwargs) -> bool:
        """
        This method updates the plot figure of each episode. When the figure is
        detected to be an embedded figure, this method will only set up the
        necessary data of the figure.
        """
        try:
            if self._reward_window != 0 and len(self._reward_history) >= self._reward_window:
                self._reward_plot.set_data(list(range(self._reward_window)), self._reward_history)
            else:
                self._reward_plot.set_data(list(range(len(self._reward_history))), self._reward_history)
            if self._reward_trend:
                self._plot_reward_trend.set_data(list(range(len(self._reward_history))), np.convolve(
                self._reward_history, np.ones(len(self._reward_history))/len(self._reward_history), mode = 'same'))
            p_settings.axes[-1].autoscale_view(False, True, True)
            p_settings.axes[-1].relim()


        except:
            pass

        if self._plot_level in [DoublePendulumRoot.C_PLOT_DEPTH_ENV, DoublePendulumRoot.C_PLOT_DEPTH_ALL]:
            DoublePendulumSystemRoot._update_plot_2d(self, p_settings = p_settings)

        return True





## ------------------------------------------------------------------------------------------------------
## ------------------------------------------------------------------------------------------------------
class DoublePendulumS4 (DoublePendulumRoot, DoublePendulumSystemS4):
    """
    This is the Double Pendulum Static 4 dimensional environment that inherits from the double pendulum root
    class, inheriting the dynamics and default reward strategy.

    Parameters
    ----------
    p_mode 
        Mode of environment. Possible values are Mode.C_MODE_SIM(default) or Mode.C_MODE_REAL.
    p_latency : timedelta
        Optional latency of environment. If not provided, the internal value of constant C_LATENCY is used by default.
    p_max_torque : float, optional
        Maximum torque applied to pendulum. The default is 20.
    p_l1 : float, optional
        Length of pendulum 1 in m. The default is 0.5
    p_l2 : float, optional
        Length of pendulum 2 in m. The default is 0.25
    p_m1 : float, optional
        Mass of pendulum 1 in kg. The default is 0.5
    p_m2 : float, optional
        Mass of pendulum 2 in kg. The default is 0.25
    p_init_angles: str, optional
        C_ANGLES_UP starts the pendulum in an upright position
        C_ANGLES_DOWN starts the pendulum in a downward position
        C_ANGLES_RND starts the pendulum from a random position.
    p_g : float, optional
        Gravitational acceleration. The default is 9.8
    p_history_length : int, optional
        Historical trajectory points to display. The default is 5.
    p_visualize : bool
        Boolean switch for visualisation. Default = False.
    p_plot_level : int
        Types and number of plots to be plotted. Default = ALL
        C_PLOT_DEPTH_ENV only plots the environment
        C_PLOT_DEPTH_REWARD only plots the reward
        C_PLOT_ALL plots both reward and the environment
    p_rst_balancingL
        Reward strategy to be used for the balancing region of the environment
    p_rst_swinging
        Reward strategy to be used for the swinging region of the environment
    p_reward_weights:list
        List of weights to be added to the dimensions of the state space for reward computation
    p_reward_trend:bool
        Boolean value stating whether to plot reward trend
    p_reward_window:int
        The number of latest rewards to be shown in the plot. Default is 0
    p_random_range:list
        The boundaries for state space for initialization of environment randomly
    p_balancing range:list
        The boundaries for state space of environment in balancing region
    p_break_swinging:bool
        Boolean value stating whether the environment shall be broken outside the balancing region
    p_logging
        Log level (see constants of class mlpro.bf.various.Log). Default = Log.C_LOG_WE.
    """

    C_NAME = 'DoublePendulumS4'


## ------------------------------------------------------------------------------------------------------
    def _normalize(self, p_state:list):
        """
        Method for normalizing the State values of the DP env based on MinMax normalisation based on static boundaries
        provided by MLPro.

        Parameters
        ----------
        p_state
            The state to be normalized

        Returns
        -------
        state
            Normalized state values

        """

        state = p_state
        for i,j in enumerate(self.get_state_space().get_dim_ids()):
            boundaries = self._state_space.get_dim(j).get_boundaries()
            state[i] = self._reward_weights[i]*(2 * ((state[i] - min(boundaries))
                             / (max(boundaries) - min(boundaries))) - 1)


        return state


## ------------------------------------------------------------------------------------------------------
    def _obs_to_mujoco(self, p_state):
        state = p_state.get_values().copy()
        state[2] = state[2] + state[0]
        mujoco_state = State(self.get_state_space())
        mujoco_state.set_values(state)
        return mujoco_state



## ------------------------------------------------------------------------------------------------------
## ------------------------------------------------------------------------------------------------------
class DoublePendulumS7 (DoublePendulumS4, DoublePendulumSystemS7):
    """
    This is the classic implementation of Double Pendulum with 7 dimensional state space including derived
    accelerations of both the poles and the input torque. The dynamics of the system are inherited from the Double
    Pendulum Root class.

    Parameters
    ----------
    p_mode 
        Mode of environment. Possible values are Mode.C_MODE_SIM(default) or Mode.C_MODE_REAL.
    p_latency : timedelta
        Optional latency of environment. If not provided, the internal value of constant C_LATENCY is used by default.
    p_max_torque : float, optional
        Maximum torque applied to pendulum. The default is 20.
    p_l1 : float, optional
        Length of pendulum 1 in m. The default is 0.5
    p_l2 : float, optional
        Length of pendulum 2 in m. The default is 0.25
    p_m1 : float, optional
        Mass of pendulum 1 in kg. The default is 0.5
    p_m2 : float, optional
        Mass of pendulum 2 in kg. The default is 0.25
    p_init_angles: str, optional
        C_ANGLES_UP starts the pendulum in an upright position
        C_ANGLES_DOWN starts the pendulum in a downward position
        C_ANGLES_RND starts the pendulum from a random position.
    p_g : float, optional
        Gravitational acceleration. The default is 9.8
    p_history_length : int, optional
        Historical trajectory points to display. The default is 5.
    p_visualize : bool
        Boolean switch for visualisation. Default = False.
    p_plot_level : int
        Types and number of plots to be plotted. Default = ALL
        C_PLOT_DEPTH_ENV only plots the environment
        C_PLOT_DEPTH_REWARD only plots the reward
        C_PLOT_ALL plots both reward and the environment
    p_rst_balancingL
        Reward strategy to be used for the balancing region of the environment
    p_rst_swinging
        Reward strategy to be used for the swinging region of the environment
    p_reward_weights:list
        List of weights to be added to the dimensions of the state space for reward computation
    p_reward_trend:bool
        Boolean value stating whether to plot reward trend
    p_reward_window:int
        The number of latest rewards to be shown in the plot. Default is 0
    p_random_range:list
        The boundaries for state space for initialization of environment randomly
    p_balancing range:list
        The boundaries for state space of environment in balancing region
    p_break_swinging:bool
        Boolean value stating whether the environment shall be broken outside the balancing region
    p_logging
        Log level (see constants of class mlpro.bf.various.Log). Default = Log.C_LOG_WE.
    """

    C_NAME = 'DoublePendulumS7'





## ------------------------------------------------------------------------------------------------------
## ------------------------------------------------------------------------------------------------------
class DoublePendulumOA4(OAEnvironment, DoublePendulumS4, oadp.DoublePendulumOA4):


    C_NAME = 'Double Pendulum A4'
## ------------------------------------------------------------------------------------------------------
    def __init__(self,
                   p_id = None,
                   p_name: str = None,
                   p_buffer_size: int = 0,
                   p_range_max: int = Range.C_RANGE_NONE,
                   p_autorun: int = Task.C_AUTORUN_NONE,
                   p_class_shared: Shared = None,
                   p_mode = Mode.C_MODE_SIM,
                   p_ada = True,
                   p_latency = None,
                   p_t_step: timedelta = None,
                   p_max_torque=20,
                   p_l1=1.0,
                   p_l2=1.0,
                   p_m1=1.0,
                   p_m2=1.0,
                   p_g=9.8,
                   p_init_angles=DoublePendulumSystemRoot.C_ANGLES_RND,
                   p_history_length=5,
                   p_fct_strans:FctSTrans=None,
                   p_fct_success:FctSuccess=None,
                   p_fct_broken:FctBroken=None,
                   p_fct_reward:FctReward=None,
                   p_wf:OAStreamWorkflow=None,
                   p_wf_reward:OAStreamWorkflow=None,
                   p_wf_success:OAStreamWorkflow=None,
                   p_wf_broken:OAStreamWorkflow=None,
                   p_plot_level:int=2,
                   p_rst_balancing = DoublePendulumS4.C_RST_BALANCING_002,
                   p_rst_swinging = DoublePendulumS4.C_RST_SWINGING_001,
                   p_rst_swinging_outer_pole = DoublePendulumS4.C_RST_SWINGING_OUTER_POLE_001,
                   p_reward_weights: list = None,
                   p_reward_trend: bool = False,
                   p_reward_window:int = 0,
                   p_random_range:list = None,
                   p_balancing_range:list = (-0.2,0.2),
                   p_swinging_outer_pole_range = (0.2,0.5),
                   p_break_swinging:bool = False,
                   p_mujoco_file = None,
                   p_frame_skip: int = 1,
                   p_state_mapping = None,
                   p_action_mapping = None,
                   p_camera_conf: tuple = (None, None, None),
                   p_visualize: bool = False,
                   p_logging: bool = Log.C_LOG_ALL,
                   **p_kwargs):

        oadp.DoublePendulumOA4.__init__(self,p_id = p_id,
                                       p_name=p_name,
                                       p_buffer_size = p_buffer_size,
                                       p_range_max = p_range_max,
                                       p_autorun = p_autorun,
                                       p_class_shared = p_class_shared,
                                       p_mode = p_mode,
                                       p_latency = p_latency,
                                       p_max_torque=p_max_torque,
                                       p_l1=p_l1,
                                       p_l2=p_l2,
                                       p_m1=p_m1,
                                       p_m2=p_m2,
                                       p_g=p_g,
                                       p_init_angles=p_init_angles,
                                       p_history_length=p_history_length,
                                       p_fct_strans=p_fct_strans,
                                       p_fct_success=p_fct_success,
                                       p_fct_broken=p_fct_broken,
                                       p_fct_reward=p_fct_reward,
                                       p_wf = p_wf,
                                       p_wf_success = p_wf_success,
                                       p_wf_broken = p_wf_broken,
                                       p_wf_reward = p_wf_reward,
                                       p_mujoco_file=p_mujoco_file,
                                       p_frame_skip=p_frame_skip,
                                       p_state_mapping=p_state_mapping,
                                       p_action_mapping=p_action_mapping,
                                       p_camera_conf=p_camera_conf,
                                       p_visualize=p_visualize,
                                       p_plot_level=p_plot_level,
                                       p_rst_balancing = p_rst_balancing,
                                       p_rst_swinging = p_rst_swinging,
                                       p_rst_swinging_outer_pole = p_rst_swinging_outer_pole,
                                       p_reward_weights = p_reward_weights,
                                       p_reward_trend=p_reward_trend,
                                       p_reward_window= p_reward_window,
                                       p_random_range=p_random_range,
                                       p_balancing_range = p_balancing_range,
                                       p_swinging_outer_pole_range = p_swinging_outer_pole_range,
                                       p_break_swinging = p_break_swinging,
                                       p_logging=p_logging)

        DoublePendulumS4.__init__(self,p_id = p_id,
                                       p_name=p_name,
                                       p_buffer_size = p_buffer_size,
                                       p_range_max = p_range_max,
                                       p_autorun = p_autorun,
                                       p_class_shared = p_class_shared,
                                       p_mode = p_mode,
                                       p_latency = p_latency,
                                       p_max_torque=p_max_torque,
                                       p_l1=p_l1,
                                       p_l2=p_l2,
                                       p_m1=p_m1,
                                       p_m2=p_m2,
                                       p_g=p_g,
                                       p_init_angles=p_init_angles,
                                       p_history_length=p_history_length,
                                       p_fct_strans=p_fct_strans,
                                       p_fct_success=p_fct_success,
                                       p_fct_broken=p_fct_broken,
                                       p_fct_reward=p_fct_reward,
                                       p_mujoco_file=p_mujoco_file,
                                       p_frame_skip=p_frame_skip,
                                       p_state_mapping=p_state_mapping,
                                       p_action_mapping=p_action_mapping,
                                       p_camera_conf=p_camera_conf,
                                       p_visualize=p_visualize,
                                       p_plot_level=p_plot_level,
                                       p_rst_balancing = p_rst_balancing,
                                       p_rst_swinging = p_rst_swinging,
                                       p_rst_swinging_outer_pole = p_rst_swinging_outer_pole,
                                       p_reward_weights = p_reward_weights,
                                       p_reward_trend=p_reward_trend,
                                       p_reward_window= p_reward_window,
                                       p_random_range=p_random_range,
                                       p_balancing_range = p_balancing_range,
                                       p_swinging_outer_pole_range = p_swinging_outer_pole_range,
                                       p_break_swinging = p_break_swinging,
                                       p_logging=p_logging)

        OAEnvironment.__init__(self,
                                 p_id = p_id,
                                 p_name = p_name,
                                 p_buffer_size = p_buffer_size,
                                 p_ada = p_ada,
                                 p_range_max = p_range_max,
                                 p_autorun = p_autorun,
                                 p_class_shared = p_class_shared,
                                 p_mode = p_mode,
                                 p_latency = p_latency,
                                 p_t_step = p_t_step,
                                 p_fct_strans = p_fct_strans,
                                 p_fct_reward = p_fct_reward,
                                 p_fct_success = p_fct_success,
                                 p_fct_broken = p_fct_broken,
                                 p_wf = p_wf,
                                 p_wf_success = p_wf_success,
                                 p_wf_broken = p_wf_broken,
                                 p_wf_reward = p_wf_reward,
                                 p_mujoco_file = p_mujoco_file,
                                 p_frame_skip = p_frame_skip,
                                 p_state_mapping = p_state_mapping,
                                 p_action_mapping = p_action_mapping,
                                 p_camera_conf = p_camera_conf,
                                 p_visualize = p_visualize,
                                 p_logging = p_logging,
                                 **p_kwargs)

        self._t_step = self.get_latency().seconds + self.get_latency().microseconds / 1000000

        self._state = State(self._state_space)
        self._target_state = State(self._state_space)
        self._target_state.set_values(np.zeros(self._state_space.get_num_dim()))





## ------------------------------------------------------------------------------------------------------
## ------------------------------------------------------------------------------------------------------
class DoublePendulumOA7(DoublePendulumOA4, oadp.DoublePendulumOA7):

    C_NAME = 'Double Pendulum A7'
    C_PLOT_ACTIVE = True


