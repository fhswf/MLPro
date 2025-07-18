## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - The integrative middleware framework for standardized machine learning
## -- Package : mlpro.bf.systems.pool
## -- Module  : doublependulum.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2023-03-05  0.0.0     LSB      Creation
## -- 2023-03-05  1.0.0     LSB      Release
## -- 2023-03-08  1.0.1     LSB      Refactoring for visualization
## -- 2024-12-11  1.0.2     DA       Refactoring      
## -- 2025-07-18  1.1.0     DA       Refactoring
## -------------------------------------------------------------------------------------------------

"""
Ver. 1.1.0 (2025-07-18)

The Double Pendulum System is an implementation of a classic control problem of Double Pendulum system. The
dynamics of the system are based on the `Double Pendulum <https://matplotlib.org/stable/gallery/animation/double_pendulum.html>`_  implementation by
`Matplotlib <https://matplotlib.org/>`_. The double pendulum is a system of two poles, with the inner pole
connected to a fixed point at one end and to outer pole at other end. The native implementation of Double
Pendulum consists of an input motor providing the torque in either directions to actuate the system.
"""

import random
from collections import deque
from datetime import timedelta

import numpy as np
from numpy import sin, cos
from matplotlib.patches import Arc, RegularPolygon
import scipy.integrate as integrate

from mlpro.bf.various import *
from mlpro.bf.exceptions import ParamError
from mlpro.bf.events import *
from mlpro.bf.mt import *
from mlpro.bf.ops import Mode
from mlpro.bf.plot import Figure, PlotSettings
from mlpro.bf.math import ESpace, Dimension
from mlpro.bf.streams import *
from mlpro.bf.systems import *



# Export list for public API
__all__ = [ 'DoublePendulumSystemRoot',
            'DoublePendulumSystemS4',
            'DoublePendulumSystemS7' ]




## ---------------------------------------------------------------------------------------------------------------------
## ---------------------------------------------------------------------------------------------------------------------
class DoublePendulumSystemRoot (System):
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
    p_fct_strans: FctSTrans, optional
        The custom State transition function.
    p_fct_success: FctSuccess, optional
        The custom Success Function.
    p_fct_broken: FctBroken, optional
        The custom Broken Function.
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

    C_NAME              = "DoublePendulumSystemRoot"

    C_SCIREF_TYPE       = ScientificObject.C_SCIREF_TYPE_ONLINE
    C_SCIREF_AUTHOR     = "John Hunter, Darren Dale, Eric Firing, Michael \
                                       Droettboom and the Matplotlib development team"
    C_SCIREF_TITLE      = "The Double Pendulum Problem"
    C_SCIREF_URL        = "https://matplotlib.org/stable/gallery/animation/double_pendulum.html"

    C_PLOT_ACTIVE       = True
    C_PLOT_DEFAULT_VIEW = PlotSettings.C_VIEW_2D

    C_CYCLE_LIMIT       = 0
    C_LATENCY           = timedelta(0, 0, 40000)

    C_ANGLES_UP         = 'up'
    C_ANGLES_DOWN       = 'down'
    C_ANGLES_RND        = 'random'

    C_VALID_ANGLES      = [C_ANGLES_UP, C_ANGLES_DOWN, C_ANGLES_RND]

    C_THRSH_GOAL        = 0

    C_ANI_FRAME         = 30
    C_ANI_STEP          = 0.001


## -------------------------------------------------------------------------------------------------
    def __init__ ( self,
                   p_id = None,
                   p_name : str =None,
                   p_range_max : int = Async.C_RANGE_NONE,
                   p_autorun = Task.C_AUTORUN_NONE,
                   p_class_shared = None,
                   p_mode = Mode.C_MODE_SIM,
                   p_latency = None,
                   p_t_step = None,
                   p_max_torque=20,
                   p_l1=1.0,
                   p_l2=1.0,
                   p_m1=1.0,
                   p_m2=1.0,
                   p_init_angles=C_ANGLES_RND,
                   p_g=9.8,
                   p_fct_strans: FctSTrans = None,
                   p_fct_success: FctSuccess = None,
                   p_fct_broken: FctBroken = None,
                   p_mujoco_file=None,
                   p_frame_skip=None,
                   p_state_mapping=None,
                   p_action_mapping=None,
                   p_camera_conf=None,
                   p_history_length=5,
                   p_visualize:bool=False,
                   p_random_range:list = None,
                   p_balancing_range:list = (-0.2,0.2),
                   p_swinging_outer_pole_range = (0.2,0.5),
                   p_break_swinging:bool = False,
                   p_logging=Log.C_LOG_ALL,
                   **p_kwargs):

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
        self._swinging_outer_pole_range = p_swinging_outer_pole_range
        self._swinging = (self._swinging_outer_pole_range, 180)
        self._break_swinging = p_break_swinging and p_balancing_range

        if self._init_angles not in self.C_VALID_ANGLES: raise ParamError("The initial angles are not valid")

        self._history_x = deque(maxlen=p_history_length)
        self._history_y = deque(maxlen=p_history_length)


        System.__init__(self,
                        p_id = p_id,
                        p_name = p_name,
                        p_range_max = p_range_max,
                        p_autorun = p_autorun,
                        p_class_shared = p_class_shared,
                        p_mode=p_mode,
                        p_latency=p_latency,
                        p_t_step = p_t_step,
                        p_fct_strans= p_fct_strans,
                        p_fct_broken= p_fct_broken,
                        p_fct_success= p_fct_success,
                        p_mujoco_file=p_mujoco_file,
                        p_frame_skip= p_frame_skip,
                        p_action_mapping=p_action_mapping,
                        p_state_mapping=p_state_mapping,
                        p_camera_conf=p_camera_conf,
                        p_visualize=p_visualize,
                        p_logging=p_logging,
                        **p_kwargs)

        self._t_step = self.get_latency().seconds + self.get_latency().microseconds / 1000000


        self._state = State(self._state_space)
        self._target_state = State(self._state_space)
        self._target_state.set_values(np.zeros(self._state_space.get_num_dim()))

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
        state
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

        p_settings.axes.append(p_figure.add_subplot(111,autoscale_on=False,xlim=(-self._L * 1.2, self._L * 1.2),
                                                                                   ylim=(-self._L * 1.2,self._L * 1.2)))
        p_settings.axes[0].set_aspect('equal')
        p_settings.axes[0].grid()
        p_settings.axes[0].set_title(self.C_NAME)

        self._cw_arc = Arc([0, 0], 0.5 * self._l1, 0.5 * self._l1, angle=0, theta1=0,
                              theta2=250, color='crimson')
        endX = (0.5 * self._l1 / 2) * np.cos(np.radians(0))
        endY = (0.5 * self._l1 / 2) * np.sin(np.radians(0))
        self._cw_arrow = RegularPolygon( xy = (endX, endY), 
                                         numVertices = 3, 
                                         radius = 0.5 * self._l1 / 9, 
                                         orientation = np.radians(180),
                                         color='crimson' )

        self._ccw_arc = Arc([0, 0], 0.5 * self._l1, 0.5 * self._l1, angle=70, theta1=0,
                               theta2=320, color='crimson')
        endX = (0.5 * self._l1 / 2) * np.cos(np.radians(70 + 320))
        endY = (0.5 * self._l1 / 2) * np.sin(np.radians(70 + 320))

        self._ccw_arrow = RegularPolygon( xy = (endX, endY), 
                                          numVertices = 3, 
                                          radius = 0.5 * self._l1 / 9, 
                                          orientation = np.radians(70 + 320),
                                          color='crimson' )

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


## ------------------------------------------------------------------------------------------------------
    def _update_plot_2d(self, p_settings: PlotSettings, **p_kwargs) -> bool:
        """
        This method updates the plot figure of each episode. When the figure is
        detected to be an embedded figure, this method will only set up the
        necessary data of the figure.
        """

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

        return True




## ------------------------------------------------------------------------------------------------------
## ------------------------------------------------------------------------------------------------------
class DoublePendulumSystemS4 (DoublePendulumSystemRoot):
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

    C_NAME = 'DoublePendulumSystemS4'


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
class DoublePendulumSystemS7 (DoublePendulumSystemS4):
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

    C_NAME = 'DoublePendulumSystemS7'

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
        DoublePendulumSystemS4._reset(self, p_seed=p_seed)
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
