## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - A Synoptic Framework for Standardized Machine Learning Tasks
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
## -------------------------------------------------------------------------------------------------

"""
Ver. 2.2.3 (2022-11-dd)

The Double Pendulum environment is an implementation of a classic control problem of Double Pendulum system. The
dynamics of the system are based on the `Double Pendulum <https://matplotlib.org/stable/gallery/animation/double_pendulum.html>`_  implementation by
`Matplotlib <https://matplotlib.org/>`_. The double pendulum is a system of two poles, with the inner pole
connected to a fixed point at one end and to outer pole at other end. The native implementation of Double
Pendulum consists of an input motor providing the torque in either directions to actuate the system. The figure
below shows the visualisation of MLPro's Double Pendulum environment.
"""
import random

from mlpro.rl import *
from mlpro.bf.various import *
import numpy as np
from numpy import sin, cos
import matplotlib.pyplot as plt
from matplotlib.patches import Arc, RegularPolygon
import scipy.integrate as integrate
from collections import deque





## ---------------------------------------------------------------------------------------------------------------------
## ---------------------------------------------------------------------------------------------------------------------
class DoublePendulumRoot (Environment):
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

    C_NAME              = "DoublePendulumRoot"

    C_SCIREF_TYPE       = ScientificObject.C_SCIREF_TYPE_ONLINE
    C_SCIREF_AUTHOR     = "John Hunter, Darren Dale, Eric Firing, Michael \
                                       Droettboom and the Matplotlib development team"
    C_SCIREF_TITLE      = "The Double Pendulum Problem"
    C_SCIREF_URL        = "https://matplotlib.org/stable/gallery/animation/double_pendulum.html"

    C_PLOT_ACTIVE       = True
    C_PLOT_DEFAULT_VIEW = PlotSettings.C_VIEW_2D

    C_CYCLE_LIMIT       = 0
    C_LATENCY           = timedelta(0, 0, 40000)

    C_REWARD_TYPE       = Reward.C_TYPE_OVERALL

    C_ANGLES_UP         = 'up'
    C_ANGLES_DOWN       = 'down'
    C_ANGLES_RND        = 'random'

    C_VALID_ANGLES      = [C_ANGLES_UP, C_ANGLES_DOWN, C_ANGLES_RND]

    C_THRSH_GOAL        = 0

    C_ANI_FRAME         = 30
    C_ANI_STEP          = 0.001

    C_PLOT_DEPTH_ENV    = 0
    C_PLOT_DEPTH_REWARD = 1
    C_PLOT_DEPTH_ALL    = 2
    C_VALID_DEPTH       = [C_PLOT_DEPTH_ENV, C_PLOT_DEPTH_REWARD, C_PLOT_DEPTH_ALL]

    C_RST_BALANCING_001 = 'rst_001'
    C_RST_BALANCING_002 = 'rst_002'
    C_VALID_RST_BALANCING = ['rst_001','rst_002']
    C_VALID_RST_REWARD = []
## -------------------------------------------------------------------------------------------------
    def __init__ ( self,
                   p_mode = Mode.C_MODE_SIM,
                   p_latency = None,
                   p_max_torque=20,
                   p_l1=1.0,
                   p_l2=1.0,
                   p_m1=1.0,
                   p_m2=1.0,
                   p_init_angles=C_ANGLES_RND,
                   p_g=9.8,
                   p_history_length=5,
                   p_visualize:bool=False,
                   p_plot_level:int=2,
                   p_rst_balancing = C_RST_BALANCING_002,
                   p_rst_swinging = None,
                   p_reward_trend: bool = False,
                   p_reward_window:int = 0,
                   p_random_range:list = None,
                   p_balancing_range:list = [-10,10],
                   p_break_swinging:bool = True,
                   p_logging=Log.C_LOG_ALL ):

        self._max_torque = p_max_torque

        self._l1 = p_l1
        self._l2 = p_l2
        self._L = p_l1 + p_l2
        self._m1 = p_m1
        self._m2 = p_m2
        self._M = p_m1 + p_m2
        self._g = p_g

        self._init_angles = p_init_angles
        self._random_range = p_random_range
        self._balancing_range = p_balancing_range
        self._break_swinging = p_break_swinging and p_balancing_range

        if self._init_angles not in self.C_VALID_ANGLES: raise ParamError("The initial angles are not valid")

        self._history_x = deque(maxlen=p_history_length)
        self._history_y = deque(maxlen=p_history_length)
        self._plot_level = p_plot_level


        super().__init__(p_mode=p_mode, p_visualize=p_visualize, p_logging=p_logging, p_latency=p_latency)
        self._t_step = self.get_latency().seconds + self.get_latency().microseconds / 1000000

        self._rst_balancing = p_rst_balancing
        self._rst_methods = {DoublePendulumRoot.C_RST_BALANCING_001:self.compute_reward_001,
                            DoublePendulumRoot.C_RST_BALANCING_002:self.compute_reward_002}
        if self._plot_level in [self.C_PLOT_DEPTH_REWARD, self.C_PLOT_DEPTH_ALL]:
            self._reward_history = []
            self._reward_trend = p_reward_trend
            self._reward_window = p_reward_window


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
                      p_unit='degrees', p_unit_latex='\textdegrees', p_boundaries=[-180, 180]))
        state_space.add_dim(
            Dimension(p_name_long='omega 1', p_name_short='w1', p_description='Angular Velocity of Pendulum 1',
                      p_name_latex='', p_unit='degrees/second', p_unit_latex='\textdegrees/s',p_boundaries=[-36000,
                                                                                                            36000]))
        state_space.add_dim(
            Dimension(p_name_long='theta 2', p_name_short='th2', p_description='Angle of pendulum 2', p_name_latex='',
                      p_unit='degrees', p_unit_latex='\textdegrees', p_boundaries=[-180, 180]))
        state_space.add_dim(
            Dimension(p_name_long='omega 2', p_name_short='w2', p_description='Angular Velocity of Pendulum 2',
                      p_name_latex='', p_unit='degrees/second', p_unit_latex='\textdegrees/s', p_boundaries=[-40000,
                                                                                                             40000]))
        action_space.add_dim(
            Dimension(p_name_long='torque 1', p_name_short='tau1', p_description='Applied Torque of Motor 1',
                      p_name_latex='', p_unit='Nm', p_unit_latex='Nm',p_boundaries=[-self._max_torque,
                                                                                     self._max_torque]))


        return state_space, action_space


## ------------------------------------------------------------------------------------------------------
    def _init_figure(self) -> Figure:
        figure = plt.figure(figsize=(7,6))

        try:
            figure.canvas.set_window_title('Environment - '+self.C_NAME)
        except AttributeError:
            figure.canvas.setWindowTitle('Environment - '+self.C_NAME)

        return figure


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
        if p_seed:
            random.seed(p_seed)
        if self._init_angles == self.C_ANGLES_UP:
            self._th1 = 0
            self._th2 = 0
        elif self._init_angles == self.C_ANGLES_DOWN:
            self._th1 = 180
            self._th2 = 180
        elif self._init_angles == self.C_ANGLES_RND:
            self._th1 = random.uniform(-180, 180)
            self._th2 = random.uniform(-180, 180)
            if self._random_range:
                if len(self._random_range) == 1 or isinstance(self._random_range, int):
                    self._th1 = random.uniform(-self._random_range,self._random_range)
                    self._th2 = random.uniform(-self._random_range,self._random_range)
                elif len(self._random_range) == 2:
                    self._th1 = random.uniform(self._random_range[0], self._random_range[1])
                    self._th2 = random.uniform(self._random_range[0], self._random_range[1])
            if self._balancing_range:
                if len(self._balancing_range) == 1 or isinstance(self._balancing_range, int):
                    self._th1 = random.uniform(-self._balancing_range,self._balancing_range)
                    self._th2 = random.uniform(-self._balancing_range,self._balancing_range)
                elif len(self._balancing_range) == 2:
                    self._th1 = random.uniform(self._balancing_range[0], self._balancing_range[1])
                    self._th2 = random.uniform(self._balancing_range[0], self._balancing_range[1])


        else:
            raise NotImplementedError("init_angles value must be up or down")


        self._omega1 = 0
        self._omega2 = 0

        state_ids = self._state.get_dim_ids()
        self._state.set_value(state_ids[0], (self._th1))
        self._state.set_value(state_ids[1], (self._omega1))
        self._state.set_value(state_ids[2], (self._th2))
        self._state.set_value(state_ids[3], (self._omega2))


        self._history_x.clear()
        self._history_y.clear()
        self._action_cw = False
        self._alpha = 0


## ------------------------------------------------------------------------------------------------------
    def _derivs(self, p_state, t,  p_torque):
        """
        This method is used to calculate the derivatives of the system, given the
        current states.

        Parameters
        ----------
        state : list
            list of current state elements [theta 1, omega 1, acc 1, theta 2, omega 2, acc 2]
        t : list
            current Timestep
        torque : float
            Applied torque of the motor

        Returns
        -------
        dydx : list
            The derivatives of the given state

        """

        dydx = np.zeros_like(p_state)
        dydx[0] = p_state[1]

        delta = p_state[2] - p_state[0]
        den1 = (self._m1 + self._m2) * self._l1 - self._m2 * self._l1 * cos(delta) * cos(delta)
        dydx[1] = ((self._m2 * self._l1 * p_state[1] * p_state[1] * sin(delta) * cos(delta)
                    + self._m2 * self._g * sin(p_state[2]) * cos(delta)
                    + self._m2 * self._l2 * p_state[3] * p_state[3] * sin(delta)
                    - (self._m1 + self._m2) * self._g * sin(p_state[0])-p_torque)
                   / den1)

        dydx[2] = p_state[3]

        den2 = (self._l2 / self._l1) * den1
        dydx[3] = ((- self._m2 * self._l2 * p_state[3] * p_state[3] * sin(delta) * cos(delta)
                    + (self._m1 + self._m2) * self._g * sin(p_state[0]) * cos(delta)
                    - (self._m1 + self._m2) * self._l1 * p_state[1] * p_state[1] * sin(delta)
                    - (self._m1 + self._m2) * self._g * sin(p_state[2]))
                   / den2)

        return dydx


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

        state = p_state.get_values()[0:4]
        for i in [0, 2]:
            if state[i] == 0:
                state[i] = 180
            elif state[i] != 0:
                sign = 1 if state[i] > 0 else -1
                state[i] = sign * (abs(state[i]) - 180)
        torque = p_action.get_sorted_values()[0]
        torque = np.clip(torque, -self._max_torque, self._max_torque)


        state = np.radians(state)

        if self._max_torque != 0:
            self._alpha = abs(torque) / self._max_torque
        else:
            self._alpha = 0

        self._y = integrate.odeint(self._derivs, state, np.arange(0, self._t_step, self.C_ANI_STEP), args=(torque,))
        state = self._y[-1].copy()


        self._action_cw = True if torque > 0 else False


        state = np.degrees(state)

        state_ids = self._state.get_dim_ids()

        for i in [0, 2]:
            if state[i] % 360 < 180:
                state[i] = state[i] % 360
            elif state[i] % 360 > 180:
                state[i] = state[i] % 360 - 360
            sign = 1 if state[i] > 0 else -1
            state[i] = sign * (abs(state[i]) - 180)

        current_state = State(self._state_space)
        for i in range(len(state)):
            current_state.set_value(state_ids[i], state[i])

        return current_state


## ------------------------------------------------------------------------------------------------------
    def _compute_reward(self, p_state_old, p_state_new):
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
        reward : Reward
            current calculated Reward values.
        """

        if self._rst_balancing in self.C_VALID_RST_BALANCING:
            current_reward = self._rst_methods[self._rst_balancing](p_state_old, p_state_new)

        else:
            raise AttributeError('Reward strategy does not exist.')
        if self._plot_level in [self.C_PLOT_DEPTH_REWARD, self.C_PLOT_DEPTH_ALL]:
            self._reward_history.append(current_reward.overall_reward)
        return current_reward


## ------------------------------------------------------------------------------------------------------
    def compute_reward_001(self, p_state_old:State=None, p_state_new:State=None):
        """

        Parameters
        ----------
        p_state_old
        p_state_new

        Returns
        -------

        """
        current_reward = Reward()
        state = p_state_new.get_values().copy()
        p_state_normalized = self._normalize(state)
        norm_state = State(self.get_state_space())
        norm_state.set_values(p_state_normalized)
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

        # d_max = self.get_state_space().distance(max_state, min_state)
        d = self.get_state_space().distance(norm_state, goal_state)
        # value = d_max - d
        # current_reward.set_overall_reward(value)

        if d <= self.C_THRSH_GOAL:
            current_reward.set_overall_reward(1)
        else:
            current_reward.set_overall_reward(-d)

        return current_reward


## ------------------------------------------------------------------------------------------------------
    def compute_reward_002(self, p_state_old:State=None, p_state_new:State=None):
        """

        Parameters
        ----------
        p_state_old
        p_state_new

        Returns
        -------

        """
        current_reward = Reward()
        state_new = p_state_new.get_values().copy()
        p_state_new_normalized = self._normalize(state_new)
        norm_state_new = State(self.get_state_space())
        norm_state_new.set_values(p_state_new_normalized)

        state_old = p_state_old.get_values().copy()
        p_state_old_normalized = self._normalize(state_old)
        norm_state_old = State(self.get_state_space())
        norm_state_old.set_values(p_state_old_normalized)
        goal_state = self._target_state

        d_old = abs(self.get_state_space().distance(goal_state, norm_state_old))
        d_new = abs(self.get_state_space().distance(goal_state, norm_state_new))
        d = d_old - d_new

        current_reward.set_overall_reward(d)

        return current_reward

## ------------------------------------------------------------------------------------------------------
    def _compute_broken(self, p_state: State) -> bool:
        """
        Custom method to compute broken state. In this case always returns false as the environment doesn't break
        """
        if self._break_swinging:
            if len(self._balancing_range) == 1 or isinstance(self._balancing_range, int):
                if -self._balancing_range >= p_state.get_values()[0] >= self._balancing_range:
                    return True
                if -self._balancing_range >= p_state.get_values()[2] >= self._balancing_range:
                    return True
            elif len(self._balancing_range) == 2:
                if self._balancing_range[0] >= p_state.get_values()[0] or p_state.get_values()[0] >= \
                        self._balancing_range[1]:
                    return True
                if self._balancing_range[0] >= p_state.get_values()[2] or p_state.get_values()[2] >= self._balancing_range[1]:
                    return True
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
        if d <= self.C_THRSH_GOAL: return True
        return False


## ------------------------------------------------------------------------------------------------------
    def _normalize(self, p_state:list):
        """
        Custom method to normalize the State values of the DP env based on static boundaries provided by MLPro

        Parameters
        ----------
        p_state:State
            The state to be normalized

        Returns
        -------
        state:State
            Normalized state values

        """

        raise NotImplementedError


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
        if self._plot_level in [DoublePendulumRoot.C_PLOT_DEPTH_ENV, self.C_PLOT_DEPTH_REWARD]:
            grid = p_figure.add_gridspec(1, 1)
        elif self._plot_level == DoublePendulumRoot.C_PLOT_DEPTH_ALL:
            if self._reward_trend:
                grid = p_figure.add_gridspec(1, 3)
                p_figure.set_size_inches(17, 5)
            else:
                grid = p_figure.add_gridspec(1,2)
                p_figure.set_size_inches(11, 5)

        if self._plot_level in [DoublePendulumRoot.C_PLOT_DEPTH_ENV, DoublePendulumRoot.C_PLOT_DEPTH_ALL]:
            p_settings.axes.append(p_figure.add_subplot(grid[0],autoscale_on=False,xlim=(-self._L * 1.2, self._L * 1.2),
                                                                                   ylim=(-self._L * 1.2,self._L * 1.2)))
            p_settings.axes[0].set_aspect('equal')
            p_settings.axes[0].grid()
            p_settings.axes[0].set_title(self.C_NAME)

            self._cw_arc = Arc([0, 0], 0.5 * self._l1, 0.5 * self._l1, angle=0, theta1=0,
                              theta2=250, color='crimson')
            endX = (0.5 * self._l1 / 2) * np.cos(np.radians(0))
            endY = (0.5 * self._l1 / 2) * np.sin(np.radians(0))
            self._cw_arrow = RegularPolygon((endX, endY), 3, 0.5 * self._l1 / 9, np.radians(180),
                                           color='crimson')

            self._ccw_arc = Arc([0, 0], 0.5 * self._l1, 0.5 * self._l1, angle=70, theta1=0,
                               theta2=320, color='crimson')
            endX = (0.5 * self._l1 / 2) * np.cos(np.radians(70 + 320))
            endY = (0.5 * self._l1 / 2) * np.sin(np.radians(70 + 320))
            self._ccw_arrow = RegularPolygon((endX, endY), 3, 0.5 * self._l1 / 9, np.radians(70 + 320),
                                            color='crimson')

            p_settings.axes[0].add_patch(self._cw_arc)
            p_settings.axes[0].add_patch(self._cw_arrow)
            p_settings.axes[0].add_patch(self._ccw_arc)
            p_settings.axes[0].add_patch(self._ccw_arrow)

            self._cw_arc.set_visible(False)
            self._cw_arrow.set_visible(False)
            self._ccw_arc.set_visible(False)
            self._ccw_arrow.set_visible(False)

            self._line, = p_settings.axes[0].plot([], [], 'o-', lw=2)
            self._trace, = p_settings.axes[0].plot([], [], '.-', lw=1, ms=2)


        if self._plot_level in [DoublePendulumRoot.C_PLOT_DEPTH_REWARD, DoublePendulumRoot.C_PLOT_DEPTH_ALL]:
            if self._reward_trend:
                p_settings.axes.append(p_figure.add_subplot(grid[1:3]))
                self._plot_reward_trend, = p_settings.axes[-1].plot(range(len(self._reward_history)),
                    self._reward_history, 'r--', lw = 2)
            else:
                p_settings.axes.append(p_figure.add_subplot(grid[1:3]))


            p_settings.axes[-1].set_title('Reward - '+ self.C_NAME)
            p_settings.axes[-1].autoscale_view()
            p_settings.axes[-1].grid()
            p_settings.axes[-1].set_xlabel('Cycle ID')
            p_settings.axes[-1].set_ylabel('Reward')
            self._reward_plot,  = p_settings.axes[-1].plot(range(len(self._reward_history)), self._reward_history,'b', lw = 1)


    ## ------------------------------------------------------------------------------------------------------
    def _update_plot_2d(self, p_settings: PlotSettings, **p_kwargs):
        """
        This method updates the plot figure of each episode. When the figure is
        detected to be an embedded figure, this method will only set up the
        necessary data of the figure.
        """
        try:
            if self._reward_window != 0 and len(self._reward_history) >= self._reward_window:
                self._reward_plot.set_data(list(range(self._reward_window)), self._reward_history[-self._reward_window:])
            else:
                self._reward_plot.set_data(list(range(len(self._reward_history))), self._reward_history)
            if self._reward_trend:
                self._plot_reward_trend.set_data(list(range(len(self._reward_history))), np.convolve(
                self._reward_history, np.ones(len(self._reward_history))/len(self._reward_history), mode = 'same'))
            p_settings.axes[-1].autoscale_view(False, True, True)
            p_settings.axes[-1].relim()


        except:
            pass
        try:
            x1 = self._l1 * sin(self._y[:, 0])
            y1 = -self._l1 * cos(self._y[:, 0])

            x2 = self._l2 * sin(self._y[:, 2]) + x1
            y2 = -self._l2 * cos(self._y[:, 2]) + y1

            for i in range(len(self._y)):
                thisx = [0, x1[i], x2[i]]
                thisy = [0, y1[i], y2[i]]

                if i % self.C_ANI_FRAME == 0:
                    self._history_x.appendleft(thisx[2])
                    self._history_y.appendleft(thisy[2])
                    self._line.set_data(thisx, thisy)
                    self._trace.set_data(self._history_x, self._history_y)

                if self._action_cw:
                    self._cw_arc.set_visible(True)
                    self._cw_arrow.set_visible(True)
                    self._cw_arc.set_alpha(self._alpha)
                    self._cw_arrow.set_alpha(self._alpha)
                    self._ccw_arc.set_visible(False)
                    self._ccw_arrow.set_visible(False)
                else:
                    self._cw_arc.set_visible(False)
                    self._cw_arrow.set_visible(False)
                    self._ccw_arc.set_visible(True)
                    self._ccw_arrow.set_visible(True)
                    self._ccw_arc.set_alpha(self._alpha)
                    self._ccw_arrow.set_alpha(self._alpha)
        except: pass




## ------------------------------------------------------------------------------------------------------
## ------------------------------------------------------------------------------------------------------
class DoublePendulumS4 (DoublePendulumRoot):
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
    def __init__ ( self, 
                   p_mode = Mode.C_MODE_SIM, 
                   p_latency = None,
                   p_max_torque=20,
                   p_l1=1.0, 
                   p_l2=1.0, 
                   p_m1=1.0, 
                   p_m2=1.0, 
                   p_init_angles=DoublePendulumRoot.C_ANGLES_RND,
                   p_g=9.8, 
                   p_history_length=5, 
                   p_visualize:bool=False,
                   p_plot_level:int = 2,
                   p_rst_balancing=DoublePendulumRoot.C_RST_BALANCING_002,
                   p_rst_swinging=None,
                   p_reward_trend: bool = False,
                   p_reward_window: int = 0,
                   p_random_range: list = None,
                   p_balancing_range: list = None,
                   p_break_swinging=True,
                   p_logging=Log.C_LOG_ALL ):

        super().__init__( p_mode=p_mode,
                          p_latency=p_latency,
                          p_max_torque=p_max_torque,
                          p_l1=p_l1,
                          p_l2=p_l2,
                          p_m1=p_m1,
                          p_m2=p_m2,
                          p_init_angles=p_init_angles,
                          p_g=p_g,
                          p_history_length=p_history_length,
                          p_visualize=p_visualize,
                          p_plot_level = p_plot_level,
                          p_rst_balancing = p_rst_balancing,
                          p_rst_swinging = p_rst_swinging,
                          p_reward_trend = p_reward_trend,
                          p_reward_window =p_reward_window,
                          p_random_range=p_random_range,
                          p_balancing_range=p_balancing_range,
                          p_break_swinging=p_break_swinging,
                          p_logging=p_logging)

        self._target_state = State(self._state_space)
        self._target_state.set_values(np.zeros(self._state_space.get_num_dim()))


## ------------------------------------------------------------------------------------------------------
    def _normalize(self, p_state:list):
        """
        Method for normalizing the State values of the DP env based on MinMax normalisation based on static boundaries
        provided by MLPro.

        Parameters
        ----------
        p_state:State
            The state to be normalized

        Returns
        -------
        state:State
            Normalized state values

        """

        state = p_state
        for i,j in enumerate(self.get_state_space().get_dim_ids()):
            boundaries = self._state_space.get_dim(j).get_boundaries()
            state[i] = (2 * ((state[i] - min(boundaries))
                             / (max(boundaries) - min(boundaries))) - 1)


        return state





## ------------------------------------------------------------------------------------------------------
## ------------------------------------------------------------------------------------------------------
class DoublePendulumS7 (DoublePendulumS4):
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
    p_logging
        Log level (see constants of class mlpro.bf.various.Log). Default = Log.C_LOG_WE.

    """

    C_NAME = 'DoublePendulumS7'

## -----------------------------------------------------------------------------------------------------
    def setup_spaces(self):
        """
        Method to set up the state and action spaces of the classic Double Pendulum Environment. Inheriting from the
        root class, this method adds 3 dimensions for accelerations and torque respectively.
        """

        state_space, action_space = super().setup_spaces()
        state_space.add_dim(
            Dimension(p_name_long='alpha 1', p_name_short='a1', p_description='Angular Acceleration of Pendulum 1',
                      p_name_latex='',p_unit='degrees/second^2', p_unit_latex='\text/s^2', p_boundaries=[-8500000,
                                                                                                         8500000]))

        state_space.add_dim(
            Dimension(p_name_long='alpha 2', p_name_short='a2', p_description='Angular Acceleration of Pendulum 2',
                      p_name_latex='',p_unit='degrees/second^2', p_unit_latex='\text/s^2', p_boundaries=[-13000000,
                                                                                                         13000000]))

        state_space.add_dim(
            Dimension(p_name_long='torque', p_name_short='tau', p_description='input torque',
                      p_name_latex='', p_unit='Newton times meters', p_unit_latex='\tNm',
                      p_boundaries=[-self._max_torque, self._max_torque]))

        return state_space, action_space


## ------------------------------------------------------------------------------------------------------
    def _reset(self, p_seed=None) -> None:
        """
        This method is used to reset the environment.

        Parameters
        ----------
        p_seed : int, optional
            The default is None.

        """
        super()._reset(p_seed=p_seed)
        self._alpha1 = 0
        self._alpha2 = 0
        for i in self._state_space.get_dim_ids()[-2:-4:-1]:
            self._state.set_value(i, 0)


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
            current_state
                Current states after simulating the latest action.

        """

        torque = p_action.get_sorted_values()[0]

        state = super()._simulate_reaction(p_state, p_action).get_values()

        for i in [0, 2]:
            if state[i] == 0:
                state[i] = 180
            elif state[i] != 0:
                sign = 1 if state[i] > 0 else -1
                state[i] = sign * (abs(state[i]) - 180)

        state = list(np.radians(state))
        delta = state[2] - state[0]

        den1 = (self._m1 + self._m2) * self._l1 - self._m2 * self._l1 * cos(delta) * cos(delta)
        state[4] = ((self._m2 * self._l1 * state[1] * state[1] * sin(delta) * cos(delta)
                     + self._m2 * self._g * sin(state[2]) * cos(delta)
                     + self._m2 * self._l2 * state[3] * state[3] * sin(delta)
                     - (self._m1 + self._m2) * self._g * sin(state[0]) - torque)
                    / den1)

        den3 = (self._l2 / self._l1) * den1
        state[5] = ((- self._m2 * self._l2 * state[3] * state[3] * sin(delta) * cos(delta)
                     + (self._m1 + self._m2) * self._g * sin(state[0]) * cos(delta)
                     - (self._m1 + self._m2) * self._l1 * state[1] * state[1] * sin(delta)
                     - (self._m1 + self._m2) * self._g * sin(state[2]))
                    / den3)

        state = np.degrees(state)

        for i in [0, 2]:
            if state[i] % 360 < 180:
                state[i] = state[i] % 360
            elif state[i] % 360 > 180:
                state[i] = state[i] % 360 - 360
            sign = 1 if state[i] > 0 else -1
            state[i] = sign * (abs(state[i]) - 180)
        state[-1] = torque
        state_ids = self._state_space.get_dim_ids()
        current_state = State(self._state_space)
        for i in range(len(state)):
            current_state.set_value(state_ids[i], state[i])

        return current_state
