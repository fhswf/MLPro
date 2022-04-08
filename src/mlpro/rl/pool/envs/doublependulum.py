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
## -------------------------------------------------------------------------------------------------

"""
Ver. 1.1.0 (2022-04-08)

This module provides an RL environment of double pendulum.
"""

from mlpro.rl.models import *
from mlpro.bf.various import *
import numpy as np
from numpy import sin, cos
import matplotlib.pyplot as plt
from matplotlib.patches import Arc, RegularPolygon
import scipy.integrate as integrate
from collections import deque


## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class DoublePendulum(Environment):
    """
    This is the main class of the Double Pendulum environment that inherits 
    Environment class from MLPro.
    
    Parameters
    ----------
    p_logging : Log, optional
        Logging functionalities. The default is Log.C_LOG_ALL.
    t_step : float, optional
        Time for each time step (in seconds). The default is 0.0025.
    t_act : int, optional
        Action frequency (with respect to the time step). The default is 20.
    max_torque : float, optional
        Maximum torque applied to pendulum 1. The default is 20.
    max_speed : float, optional
        Maximum speed applied to pendulum 1. The default is 10.
    l1 : float, optional
        Length of pendulum 1 in m. The default is 0.5
    l2 : float, optional
        Length of pendulum 2 in m. The default is 0.25
    m1 : float, optional
        Mass of pendulum 1 in kg. The default is 0.5
    m2 : float, optional
        Mass of pendulum 2 in kg. The default is 0.25
    init_angles: str, optional
        'up' starts the pendulum in an upright position
        'down' starts the pendulum in a downward position
        'random' starts the pendulum from a random position.
    g : float, optional
        Gravitational acceleration. The default is 9.8
    history_length : int, optional
        Historical trajectory points to display. The default is 2.
        
    Attributes
    ----------
    C_NAME : str
        Name of the environment.
    C_CYCLE_LIMIT : int
        The number of cycle limit.
    C_LATENCY : timedelta()
        Latency.
    C_REWARD_TYPE : Reward
        Rewarding type.
    """
    C_NAME = "DoublePendulum"
    C_CYCLE_LIMIT = 0
    C_LATENCY = timedelta(0, 0, 0)
    C_REWARD_TYPE = Reward.C_TYPE_OVERALL

    ## -------------------------------------------------------------------------------------------------
    def __init__(self, p_logging=Log.C_LOG_ALL, t_step=0.0025, t_act=20, max_torque=20,
                 max_speed=10, l1=0.5, l2=0.25, m1=0.5, m2=0.25, init_angles='down', 
                 g=9.8, history_length=3):
        self.t_step = t_step
        self.t_act = t_act

        self.set_latency(timedelta(0, t_act * t_step, 0))

        self.max_torque = max_torque
        self.max_speed = max_speed

        self.l1 = l1
        self.l2 = l2
        self.L = l1 + l2
        self.m1 = m1
        self.m2 = m2
        self.M = m1+m2
        self.g = g
        
        self.th1dot = 0
        self.th2dot = 0
        
        if init_angles=='up':
            self.th1 = 180
            self.th2 = 180
        elif init_angles=='down':
            self.th1 = 0
            self.th2 = 0
        elif init_angles=='random':
            self.th1 = np.random.rand(1)[0]*180
            self.th2 = np.random.rand(1)[0]*180
        else:
            raise NotImplementedError("init_angles value must be up, down, or random")
        
        self.history_x = deque(maxlen=history_length)
        self.history_y = deque(maxlen=history_length)

        super().__init__(p_mode=Environment.C_MODE_SIM, p_logging=p_logging)

        self.C_SCIREF_TYPE = self.C_SCIREF_TYPE_ONLINE
        self.C_SCIREF_AUTHOR = "John Hunter, Darren Dale, Eric Firing, Michael \
                                Droettboom and the Matplotlib development team"
        self.C_SCIREF_TITLE = "The Double Pendulum Problem"
        self.C_SCIREF_URL = "https://matplotlib.org/stable/gallery/animation/double_pendulum.html"

        self._state = State(self._state_space)

        self.reset()

    ## -------------------------------------------------------------------------------------------------
    def setup_spaces(self):
        """
        This method is used to set up action and state spaces of the system.

        Returns
        -------
        state_space : ESpace()
            State space of the system.
        action_space : ESpace()
            Action space of the system.

        """
        state_space = ESpace()
        action_space = ESpace()

        state_space.add_dim(Dimension('theta 1', 'th1', 'Angle of Pendulum 1', '', 'degrees',
                                      '\textdegrees', [-np.pi, np.pi]))
        state_space.add_dim(Dimension('omega 1', 'w1', 'Angular Velocity of Pendulum 1', '',
                                      'degrees/second', '\textdegrees/s', [-np.inf, np.inf]))
        state_space.add_dim(Dimension('theta 2', 'th2', 'Angle of pendulum 2', '', 'degrees',
                                      '\textdegrees', [-np.pi, np.pi]))
        state_space.add_dim(Dimension('omega 2', 'w2', 'Angular Velocity of Pendulum 2', '',
                                      'degrees/second', '\textdegrees/s', [-np.inf, np.inf]))

        action_space.add_dim(Dimension('torque 1', 'tau1', 'Applied Torque of Motor 1', '',
                                       'Nm', 'Nm', [-self.max_torque, self.max_torque]))

        return state_space, action_space

    ## -------------------------------------------------------------------------------------------------
    def derivs(self, state, t=None):
        """
        This method is used to calculate the derivatives of the system, given the
        current states.

        Parameters
        ----------
        state : list
            [th, th1dot, th2, th2dot]

        t : list
            Timestep

        Returns
        -------
        dydx : list
            The derivatives of the given state

        """
        dydx = np.zeros_like(state)
        dydx[0] = state[1]

        delta = state[2] - state[0]
        den1 = (self.m1 + self.m2) * self.l1 - self.m2 * self.l1 * cos(delta) * cos(delta)
        dydx[1] = ((self.m2 * self.l1 * state[1] * state[1] * sin(delta) * cos(delta)
                    + self.m2 * self.g * sin(state[2]) * cos(delta)
                    + self.m2 * self.l2 * state[3] * state[3] * sin(delta)
                    - (self.m1 + self.m2) * self.g * sin(state[0]))
                   / den1)

        dydx[2] = state[3]

        den2 = (self.l2 / self.l1) * den1
        dydx[3] = ((- self.m2 * self.l2 * state[3] * state[3] * sin(delta) * cos(delta)
                    + (self.m1 + self.m2) * self.g * sin(state[0]) * cos(delta)
                    - (self.m1 + self.m2) * self.l1 * state[1] * state[1] * sin(delta)
                    - (self.m1 + self.m2) * self.g * sin(state[2]))
                   / den2)

        return dydx

    ## -------------------------------------------------------------------------------------------------
    @staticmethod
    def angle_normalize(x):
        """
        This method is called to ensure a normalized angle in radians.

        Returns
        -------
        angle : float
            Normalized angle.

        """
        return ((x + np.pi) % (2 * np.pi)) - np.pi

    ## -------------------------------------------------------------------------------------------------
    def _reset(self, p_seed=None) -> None:
        """
        This method is used to reset the environment.

        Parameters
        ----------
        p_seed : int, optional
            Not yet implemented. The default is None.

        """
        state_ids = self._state.get_dim_ids()
        self._state.set_value(state_ids[0], np.radians(self.th1))
        self._state.set_value(state_ids[1], np.radians(self.th1dot))
        self._state.set_value(state_ids[2], np.radians(self.th2))
        self._state.set_value(state_ids[3], np.radians(self.th2dot))

        self.history_x.clear()
        self.history_y.clear()
        self.action_cw = False
        self.alpha = 0

    ## -------------------------------------------------------------------------------------------------
    def _simulate_reaction(self, p_state: State, p_action: Action) -> State:
        """
        This method is used to calculate the next states of the system after a set of actions.

        Parameters
        ----------
        p_state : State
            State.
        p_action : Action
            Action.

        Returns
        -------
        _state : State
            Current states.

        """
        state = p_state.get_values()
        th1, th1dot, th2, th2dot = state
        torque = p_action.get_sorted_values()[0]
        torque = np.clip(torque, -self.max_torque, self.max_torque)
        
        self.alpha = abs(torque)/self.max_torque
        
        state[1] = th1dot + (3 * self.g / (2 * self.l1) * sin(th1) + 3.0 /
                             (self.m1 * self.l1 ** 2) * torque) * self.t_step

        if abs(th1dot) > self.max_speed:
            state[1] = np.clip(state[1], -th1dot, th1dot)
        else:
            state[1] = np.clip(state[1], -self.max_speed, self.max_speed)

        self.y = integrate.odeint(self.derivs, state, np.arange(0, self.t_act * self.t_step, self.t_step))
        state = self.y[-1]
        state[0] = DoublePendulum.angle_normalize(state[0])
        state[2] = DoublePendulum.angle_normalize(state[2])
        
        self.action_cw = True if torque <= 0 else False
        state_ids = self._state.get_dim_ids()
        for i in range(len(state)):
            self._state.set_value(state_ids[i], state[i])

        return self._state

    ## -------------------------------------------------------------------------------------------------
    def _compute_broken(self, p_state: State) -> bool:
        """
        This method computes the broken flag. This method can be redefined.

        Parameters
        ----------
        p_state : State
            State.

        Returns
        -------
        bool :
            Broken or not.

        """

        return False

    ## -------------------------------------------------------------------------------------------------
    def _compute_success(self, p_state: State) -> bool:
        """
        This method computes the success flag. This method can be redefined.

        Parameters
        ----------
        p_state : State
            State.

        Returns
        -------
        bool :
            Success or not success.

        """

        return False

    ## -------------------------------------------------------------------------------------------------
    def _compute_reward(self, p_state_old: State, p_state_new: State) -> Reward:
        """
        This method calculates the reward for C_TYPE_OVERALL reward type.

        Parameters
        ----------
        p_state_old : State
            Previous state.
        p_state_new : State
            New state.

        Returns
        -------
        reward : Reward
            Reward values.

        """
        reward = Reward(Reward.C_TYPE_OVERALL)
        
        target = np.array([np.pi, 0.0, np.pi, 0.0])
        state = p_state_new.get_values()
        old_state = p_state_old.get_values()
        
        th1_count = 0
        for th1 in self.y[::-1, 0]:
            ang = np.degrees(DoublePendulum.angle_normalize(th1))
            if ang > 170 or ang < 190 or \
                    ang < -170 or ang > -190:
                th1_count += 1
            else:
                break
        th1_distance = np.pi - abs(DoublePendulum.angle_normalize(np.radians(state[0])))
        th1_distance_costs = 4 if th1_distance <= 0.1 else 0.3 / th1_distance
        
        th1_speed_costs = np.pi * abs(state[1]) / self.max_speed
        
        # max acceleration in one timestep is assumed to be double the max speed
        th1_acceleration_costs = np.pi * abs(self.y[-1, 1]-self.y[-2, 1]) / (2 * self.max_speed)
        
        inner_pole_costs = (th1_distance_costs * th1_count / len(self.y)) - th1_speed_costs - (th1_acceleration_costs ** 0.5)
        inner_pole_weight = (self.l1/2)*self.m1
        
        th2_count = 0
        for th2 in self.y[::-1, 2]:
            ang = np.degrees(DoublePendulum.angle_normalize(th2))
            if ang > 170 or ang < 190 or \
                    ang < -170 or ang > -190:
                th2_count += 1
            else:
                break
        th2_distance = np.pi - abs(DoublePendulum.angle_normalize(np.radians(state[2])))
        th2_distance_costs = 4 if th2_distance <= 0.1 else 0.3 / th2_distance
        
        th2_speed_costs = np.pi * abs(state[3]) / self.max_speed
        
        th2_acceleration_costs = np.pi * abs(self.y[-1, 3]-self.y[-2, 3]) / (2 * self.max_speed)
        
        outer_pole_costs = (th2_distance_costs * th2_count / len(self.y)) - th2_speed_costs - (th2_acceleration_costs ** 0.5)
        outer_pole_weight = 0.5 * (self.l2/2)*self.m2
        
        change_costs = ((np.linalg.norm(target[::2] - np.array(old_state)[::2])*inner_pole_weight) - 
                        (np.linalg.norm(target[::2] - np.array(state)[::2])*outer_pole_weight))
        
        reward.set_overall_reward((inner_pole_costs * inner_pole_weight) + (outer_pole_costs * outer_pole_weight) )
                                  # - (self.alpha * np.pi/2) + (change_costs))

        return reward

    ## -------------------------------------------------------------------------------------------------
    def init_plot(self, p_figure=None):
        """
        This method initializes the plot figure of each episode. When the environment
        is reset, the previous figure is closed and reinitialized. 
        
        Parameters
        ----------
        p_figure : matplotlib.figure.Figure
            A Figure object of the matplotlib library.
        """
        if hasattr(self, 'fig'):
            plt.close(self.fig)

        if p_figure is None:
            self.fig = plt.figure(figsize=(5, 4))
            self.embedded_fig = False
        else:
            self.fig = p_figure
            self.embedded_fig = True

        self.ax = self.fig.add_subplot(autoscale_on=False,
                                       xlim=(-self.L * 1.2, self.L * 1.2), ylim=(-self.L * 1.2, self.L * 1.2))
        self.ax.set_aspect('equal')
        self.ax.grid()

        self.cw_arc = Arc([0, 0], 0.5 * self.l1, 0.5 * self.l1, angle=0, theta1=0,
                          theta2=250, color='crimson')
        endX = (0.5 * self.l1 / 2) * np.cos(np.radians(0))
        endY = (0.5 * self.l1 / 2) * np.sin(np.radians(0))
        self.cw_arrow = RegularPolygon((endX, endY), 3, 0.5 * self.l1 / 9, np.radians(180),
                                       color='crimson')

        self.ccw_arc = Arc([0, 0], 0.5 * self.l1, 0.5 * self.l1, angle=70, theta1=0,
                           theta2=320, color='crimson')
        endX = (0.5 * self.l1 / 2) * np.cos(np.radians(70 + 320))
        endY = (0.5 * self.l1 / 2) * np.sin(np.radians(70 + 320))
        self.ccw_arrow = RegularPolygon((endX, endY), 3, 0.5 * self.l1 / 9, np.radians(70 + 320),
                                        color='crimson')

        self.ax.add_patch(self.cw_arc)
        self.ax.add_patch(self.cw_arrow)
        self.ax.add_patch(self.ccw_arc)
        self.ax.add_patch(self.ccw_arrow)

        self.cw_arc.set_visible(False)
        self.cw_arrow.set_visible(False)
        self.ccw_arc.set_visible(False)
        self.ccw_arrow.set_visible(False)

        self.line, = self.ax.plot([], [], 'o-', lw=2)
        self.trace, = self.ax.plot([], [], '.-', lw=1, ms=2)

    ## -------------------------------------------------------------------------------------------------
    def update_plot(self):
        """
        This method updates the plot figure of each episode. When the figure is
        detected to be an embedded figure, this method will only set up the 
        necessary data of the figure.
        
        """
        x1 = self.l1 * sin(self.y[:, 0])
        y1 = -self.l1 * cos(self.y[:, 0])

        x2 = self.l2 * sin(self.y[:, 2]) + x1
        y2 = -self.l2 * cos(self.y[:, 2]) + y1

        thisx = [0, x1[-1], x2[-1]]
        thisy = [0, y1[-1], y2[-1]]
        self.history_x.appendleft(thisx[2])
        self.history_y.appendleft(thisy[2])
        self.line.set_data(thisx, thisy)
        self.trace.set_data(self.history_x, self.history_y)

        if self.action_cw:
            self.cw_arc.set_visible(True)
            self.cw_arrow.set_visible(True)
            self.cw_arc.set_alpha(self.alpha)
            self.cw_arrow.set_alpha(self.alpha)
            self.ccw_arc.set_visible(False)
            self.ccw_arrow.set_visible(False)
        else:
            self.cw_arc.set_visible(False)
            self.cw_arrow.set_visible(False)
            self.ccw_arc.set_visible(True)
            self.ccw_arrow.set_visible(True)
            self.ccw_arc.set_alpha(self.alpha)
            self.ccw_arrow.set_alpha(self.alpha)

        if not self.embedded_fig:
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()
