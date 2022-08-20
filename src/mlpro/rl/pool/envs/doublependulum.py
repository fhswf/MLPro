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
## -------------------------------------------------------------------------------------------------

"""
Ver. 1.4.7 (2022-08-20)

The Double Pendulum environment is an implementation of a classic control problem of Double Pendulum system. The
dynamics of the system are based on the `Double Pendulum <https://matplotlib.org/stable/gallery/animation/double_pendulum.html>`_  implementation by
`Matplotlib <https://matplotlib.org/>`_. The double pendulum is a system of two poles, with the inner pole
connected to a fixed point at one end and to outer pole at other end. The native implementation of Double
Pendulum consists of an input motor providing the torque in either directions to actuate the system. The figure
below shows the visualisation of MLPro's Double Pendulum environment.
"""

from mlpro.rl.models import *
from mlpro.bf.various import *
import numpy as np
from numpy import sin, cos
import matplotlib.pyplot as plt
plt.ion()
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
    def __init__(self, p_logging=Log.C_LOG_ALL, t_step=0.2, t_act=5, max_torque=1,
                 max_speed=10, l1=1.0, l2=1.0, m1=1.0, m2=1.0, init_angles='down',
                 g=9.8, history_length=2):
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

        self.init_angles = init_angles

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

        state_space.add_dim(Dimension(p_name_long='theta 1', p_name_short='th1', p_description='Angle of Pendulum 1', p_name_latex='', p_unit='degrees',
                                      p_unit_latex='\textdegrees', p_boundaries=[-180,180]))
        state_space.add_dim(Dimension(p_name_long='omega 1', p_name_short='w1', p_description='Angular Velocity of Pendulum 1', p_name_latex='',
                                      p_unit='degrees/second', p_unit_latex='\textdegrees/s', p_boundaries=[-796.617, 559.5576]))
        state_space.add_dim(Dimension(p_name_long='acc 1', p_name_short='a1', p_description='Angular Acceleration of Pendulum 1', p_name_latex='',
                                      p_unit='degrees/second^2', p_unit_latex='\text/s^2', p_boundaries=[-6732.31, 5870.988]))
        state_space.add_dim(Dimension(p_name_long='theta 2', p_name_short='th2', p_description='Angle of pendulum 2', p_name_latex='',p_unit= 'degrees',
                                      p_unit_latex='\textdegrees', p_boundaries=[-180, 180]))
        state_space.add_dim(Dimension(p_name_long='omega 2', p_name_short='w2', p_description='Angular Velocity of Pendulum 2', p_name_latex='',
                                      p_unit='degrees/second', p_unit_latex='\textdegrees/s', p_boundaries=[-904.93, 844.5236]))
        state_space.add_dim(Dimension(p_name_long='acc 2', p_name_short='a2', p_description='Angular Acceleration of Pendulum 2', p_name_latex='',
                                      p_unit='degrees/second^2', p_unit_latex='\text/s^2', p_boundaries=[-9650.26, 6805.587]))

        action_space.add_dim(Dimension(p_name_long='torque 1', p_name_short='tau1', p_description='Applied Torque of Motor 1', p_name_latex='',
                                       p_unit='Nm', p_unit_latex='Nm', p_boundaries=[-self.max_torque, self.max_torque]))

        return state_space, action_space

    ## -------------------------------------------------------------------------------------------------
    def derivs(self, state, t, torque):
        """
        This method is used to calculate the derivatives of the system, given the
        current states.

        Parameters
        ----------
        state : list
            [theta 1, omega 1, acc 1, theta 2, omega 2, acc 2]
        t : list
            Timestep
        torque : float
            Applied torque of the motor

        Returns
        -------
        dydx : list
            The derivatives of the given state

        """
        dydx = np.zeros_like(state)
        dydx[0] = state[1]

        delta = state[3] - state[0]
        den1 = (self.m1 + self.m2) * self.l1 - self.m2 * self.l1 * cos(delta) * cos(delta)
        dydx[1] = ((self.m2 * self.l1 * state[1] * state[1] * sin(delta) * cos(delta)
                    + self.m2 * self.g * sin(state[3]) * cos(delta)
                    + self.m2 * self.l2 * state[4] * state[4] * sin(delta)
                    - (self.m1 + self.m2) * self.g * sin(state[0])-torque)
                   / den1)

        dydx[3] = state[4]

        den3 = (self.l2 / self.l1) * den1
        dydx[4] = ((- self.m2 * self.l2 * state[4] * state[4] * sin(delta) * cos(delta)
                    + (self.m1 + self.m2) * self.g * sin(state[0]) * cos(delta)
                    - (self.m1 + self.m2) * self.l1 * state[1] * state[1] * sin(delta)
                    - (self.m1 + self.m2) * self.g * sin(state[3]))
                   / den3)

        return dydx

    ## -------------------------------------------------------------------------------------------------
    def _data_normalization(self, p_value, p_boundaries):
        """
        This method is called to normalize any data in between -1 to 1 by considering their boundaries.
        If the boundaries are infinity, then the data is not normalized.

        Parameters
        ----------
        p_value : float
            Input values.
        p_boundaries : Array
            The min-max boundaries of the parameter, e.g. [min, max]

        Returns
        -------
        normalized_value: float

        """
        if p_boundaries[0] == -np.inf or p_boundaries[0] == np.inf:
            return p_value
        elif p_boundaries[1] == -np.inf or p_boundaries[1] == np.inf:
            return p_value
        else:
            normalized_value = (2*((p_value-min(p_boundaries))/(max(p_boundaries)-min(p_boundaries)))-1)
            return normalized_value

    ## -------------------------------------------------------------------------------------------------
    def _reset(self, p_seed=None) -> None:
        """
        This method is used to reset the environment.

        Parameters
        ----------
        p_seed : int, optional
            Not yet implemented. The default is None.

        """
        if self.init_angles =='up':
            self.th1 = 0
            self.th2 = 0
        elif self.init_angles=='down':
            self.th1 = 180
            self.th2 = 180
        elif self.init_angles=='random':
            self.th1 = np.random.rand(1)[0]*180
            self.th2 = np.random.rand(1)[0]*180
        else:
            raise NotImplementedError("init_angles value must be up or down")

        self.a1 = 0
        self.a2 = 0

        self.th1dot = 0
        self.th2dot = 0

        state_ids = self._state.get_dim_ids()
        self._state.set_value(state_ids[0], (self.th1))
        self._state.set_value(state_ids[1], (self.th1dot))
        self._state.set_value(state_ids[2], (self.a1))
        self._state.set_value(state_ids[3], (self.th2))
        self._state.set_value(state_ids[4], (self.th2dot))
        self._state.set_value(state_ids[5], (self.a2))

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
        for i in [0, 3]:
            if state[i] == 0:
                state[i] = 180
            elif state[i] != 0:
                # state[i] = list(range(-180, 180))[int(-state[i])]
                if state[i] > 0:
                    sign = 1
                else:
                    sign = -1
                state[i] = sign*(abs(state[i]) - 180)

        th1, th1dot, a1, th2, th2dot, a2 = np.radians(state)
        state = [th1, th1dot, 0, th2, th2dot, 0]
        torque = p_action.get_sorted_values()[0]
        torque = np.clip(torque, -self.max_torque, self.max_torque)
        torque = tuple(torque.reshape([1]))


        if self.max_torque != 0:
            self.alpha = abs(torque[0])/self.max_torque
        else: self.alpha = 0


        self.y = integrate.odeint(self.derivs, state, np.arange(0, self.t_step/self.t_act, 0.001), args=(torque,))
        state = self.y[-1].copy()

        delta = state[3]-state[0]

        den1 = (self.m1 + self.m2) * self.l1 - self.m2 * self.l1 * cos(delta) * cos(delta)
        state[2]= ((self.m2 * self.l1 * state[1] * state[1] * sin(delta) * cos(delta)
                    + self.m2 * self.g * sin(state[3]) * cos(delta)
                    + self.m2 * self.l2 * state[4] * state[4] * sin(delta)
                    - (self.m1 + self.m2) * self.g * sin(state[0])-torque)
                   / den1)

        den3 = (self.l2 / self.l1) * den1
        state[5] = ((- self.m2 * self.l2 * state[4] * state[4] * sin(delta) * cos(delta)
                    + (self.m1 + self.m2) * self.g * sin(state[0]) * cos(delta)
                    - (self.m1 + self.m2) * self.l1 * state[1] * state[1] * sin(delta)
                    - (self.m1 + self.m2) * self.g * sin(state[3]))
                   / den3)

        state = np.degrees(state)
        self.action_cw = True if torque[0] > 0 else False
        state_ids = self._state.get_dim_ids()

        for i in [0,3]:
            if state[i] % 360 < 180:
                state[i] = state[i] % 360
            elif state[i] % 360 > 180:
                state[i] = state[i] % 360 - 360
            # state[i] = list(range(-180, 180))[int(-state[i])]
            if state[i] > 0:
                sign = 1
            else:
                sign = -1
            state[i] = sign*(abs(state[i]) - 180)

        current_state = State(self._state_space)
        for i in range(len(state)):
            current_state.set_value(state_ids[i], state[i])

        return current_state

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
        state = p_state_old.get_values()
        th1, th1dot,a1, th2, th2dot,a2 = state

        self._state_space.get_dim_ids()
        id = self._state_space.get_dim_ids()[0]
        th1_boundaries = self._state_space.get_dim(id).get_boundaries()
        th1 = self._data_normalization(th1, th1_boundaries)

        self._state_space.get_dim_ids()
        id = self._state_space.get_dim_ids()[1]
        th1dot_boundaries = self._state_space.get_dim(id).get_boundaries()
        th1dot = self._data_normalization(th1dot, th1dot_boundaries)

        self._state_space.get_dim_ids()
        id = self._state_space.get_dim_ids()[2]
        a1_boundaries = self._state_space.get_dim(id).get_boundaries()
        a1 = self._data_normalization(a1, a1_boundaries)

        self._state_space.get_dim_ids()
        id = self._state_space.get_dim_ids()[3]
        th2_boundaries = self._state_space.get_dim(id).get_boundaries()
        th2 = self._data_normalization(th2, th2_boundaries)

        self._state_space.get_dim_ids()
        id = self._state_space.get_dim_ids()[4]
        th2dot_boundaries = self._state_space.get_dim(id).get_boundaries()
        th2dot = self._data_normalization(th2dot, th2dot_boundaries)

        self._state_space.get_dim_ids()
        id = self._state_space.get_dim_ids()[5]
        a2_boundaries = self._state_space.get_dim(id).get_boundaries()
        a2 = self._data_normalization(a2, a2_boundaries)

        reward = Reward(Reward.C_TYPE_OVERALL)

        target = np.array([np.pi, 0.0, np.pi, 0.0])
        state = p_state_new.get_values()
        old_state = p_state_old.get_values()

        th1_count = 0
        for th1 in self.y[::-1, 0]:
            ang = np.degrees(self._data_normalization(th1, th1_boundaries))
            if ang > 170 or ang < 190 or \
                    ang < -170 or ang > -190:
                th1_count += 1
            else:
                break
        th1_distance = np.pi - abs(self._data_normalization(th1, th1_boundaries))
        th1_distance_costs = 4 if th1_distance <= 0.1 else 0.3 / th1_distance

        th1_speed_costs = np.pi * abs(state[1]) / self.max_speed

        # max acceleration in one timestep is assumed to be double the max speed
        th1_acceleration_costs = np.pi * abs(self.y[-1, 1]-self.y[-2, 1]) / (2 * self.max_speed)

        inner_pole_costs = (th1_distance_costs * th1_count / len(self.y)) - th1_speed_costs - (th1_acceleration_costs ** 0.5)
        inner_pole_weight = (self.l1/2)*self.m1

        th2_count = 0
        for th2 in self.y[::-1, 2]:
            ang = np.degrees(self._data_normalization(th2, th2_boundaries))
            if ang > 170 or ang < 190 or \
                    ang < -170 or ang > -190:
                th2_count += 1
            else:
                break
        th2_distance = np.pi - abs(self._data_normalization(th2, th2_boundaries))
        th2_distance_costs = 4 if th2_distance <= 0.1 else 0.3 / th2_distance

        th2_speed_costs = np.pi * abs(state[3]) / self.max_speed

        th2_acceleration_costs = np.pi * abs(self.y[-1, 3]-self.y[-2, 3]) / (2 * self.max_speed)

        outer_pole_costs = (th2_distance_costs * th2_count / len(self.y)) - th2_speed_costs - (th2_acceleration_costs ** 0.5)
        outer_pole_weight = 0.5 * (self.l2/2)*self.m2

        change_costs = ((np.linalg.norm(target[::2] - np.array(old_state)[::3])*(inner_pole_weight)) -
                        (np.linalg.norm(target[::2] - np.array(state)[::3])*(outer_pole_weight)))

        reward.set_overall_reward((inner_pole_costs * inner_pole_weight) + (outer_pole_costs * outer_pole_weight) - (self.alpha * np.pi/2) + (change_costs))


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

        x2 = self.l2 * sin(self.y[:, 3]) + x1
        y2 = -self.l2 * cos(self.y[:, 3]) + y1

        # def animate(i):
        for i in range(len(self.y)):
            thisx = [0, x1[i], x2[i]]
            thisy = [0, y1[i], y2[i]]

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

            if not self.embedded_fig and i%30 ==0 :
                self.fig.canvas.draw()
                self.fig.canvas.flush_events()





## ---------------------------------------------------------------------------------------------------------------------
## ---------------------------------------------------------------------------------------------------------------------
class DoublePendulumRoot(Environment):
    """
    This is the root double pendulum environment class inherited from Environment class with four dimensional state
    space and underlying implementation of the Double Pendulum dynamics.
    """

    C_NAME = "DoublePendulumRoot"
    C_CYCLE_LIMIT = 0
    C_LATENCY = timedelta(0, 0, 0)
    C_REWARD_TYPE = Reward.C_TYPE_OVERALL


## -------------------------------------------------------------------------------------------------
    def __init__(self, p_logging=Log.C_LOG_ALL, t_step=0.2, t_act=5, max_torque=0,
                 max_speed=10, l1=1.0, l2=1.0, m1=1.0, m2=1.0, init_angles='down',
                 g=9.8, history_length=5):
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
        self.M = m1 + m2
        self.g = g

        self.init_angles = init_angles

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
                      p_name_latex='', p_unit='degrees/second', p_unit_latex='\textdegrees/s',p_boundaries=[-796.617, 559.5576]))
        state_space.add_dim(
            Dimension(p_name_long='theta 2', p_name_short='th2', p_description='Angle of pendulum 2', p_name_latex='',
                      p_unit='degrees', p_unit_latex='\textdegrees', p_boundaries=[-180, 180]))
        state_space.add_dim(
            Dimension(p_name_long='omega 2', p_name_short='w2', p_description='Angular Velocity of Pendulum 2',
                      p_name_latex='', p_unit='degrees/second', p_unit_latex='\textdegrees/s', p_boundaries=[-904.93, 844.5236]))
        action_space.add_dim(
            Dimension(p_name_long='torque 1', p_name_short='tau1', p_description='Applied Torque of Motor 1',
                      p_name_latex='', p_unit='Nm', p_unit_latex='Nm',p_boundaries=[-self.max_torque, self.max_torque]))


        return state_space, action_space


## ------------------------------------------------------------------------------------------------------
    def _reset(self, p_seed=None) -> None:
        """
        This method is used to reset the environment. The environment is reset to the initial position set during
        the initialization of the environment.

        Parameters
        ----------
        p_seed : int, optional
            Not yet implemented. The default is None.

        """
        if self.init_angles =='up':
            self.th1 = 0
            self.th2 = 0
        elif self.init_angles=='down':
            self.th1 = 180
            self.th2 = 180
        elif self.init_angles=='random':
            self.th1 = np.random.rand(1)[0]*180
            self.th2 = np.random.rand(1)[0]*180
        else:
            raise NotImplementedError("init_angles value must be up or down")


        self.th1dot = 0
        self.th2dot = 0

        state_ids = self._state.get_dim_ids()
        self._state.set_value(state_ids[0], (self.th1))
        self._state.set_value(state_ids[1], (self.th1dot))
        self._state.set_value(state_ids[2], (self.th2))
        self._state.set_value(state_ids[3], (self.th2dot))


        self.history_x.clear()
        self.history_y.clear()
        self.action_cw = False
        self.alpha = 0


    ## ------------------------------------------------------------------------------------------------------
    def derivs(self, state, t, torque):
        """
        This method is used to calculate the derivatives of the system, given the
        current states.

        Parameters
        ----------
        state : list
            [theta 1, omega 1, acc 1, theta 2, omega 2, acc 2]
        t : list
            Timestep
        torque : float
            Applied torque of the motor

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
                    - (self.m1 + self.m2) * self.g * sin(state[0])-torque)
                   / den1)

        dydx[2] = state[3]

        den2 = (self.l2 / self.l1) * den1
        dydx[3] = ((- self.m2 * self.l2 * state[3] * state[3] * sin(delta) * cos(delta)
                    + (self.m1 + self.m2) * self.g * sin(state[0]) * cos(delta)
                    - (self.m1 + self.m2) * self.l1 * state[1] * state[1] * sin(delta)
                    - (self.m1 + self.m2) * self.g * sin(state[2]))
                   / den2)

        return dydx


## ------------------------------------------------------------------------------------------------------
    def _simulate_reaction(self, p_state:State, p_action:Action):
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
        state = p_state.get_values()[0:4]
        for i in [0, 2]:
            if state[i] == 0:
                state[i] = 180
            elif state[i] != 0:
                sign = 1 if state[i] > 0 else -1
                state[i] = sign * (abs(state[i]) - 180)
        torque = p_action.get_sorted_values()[0]
        torque = np.clip(torque, -self.max_torque, self.max_torque)


        state = np.radians(state)

        if self.max_torque != 0:
            self.alpha = abs(torque) / self.max_torque
        else:
            self.alpha = 0

        self.y = integrate.odeint(self.derivs, state, np.arange(0, self.t_step / self.t_act, 0.001), args=(torque,))
        state = self.y[-1].copy()


        self.action_cw = True if torque > 0 else False


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
    def _compute_reward(self, p_state_old: State, p_state_new: State) -> Reward:
        return Reward()


## ------------------------------------------------------------------------------------------------------
    def _compute_broken(self, p_state: State) -> bool:
        return False


## ------------------------------------------------------------------------------------------------------
    def _compute_success(self, p_state):
        return False


## ------------------------------------------------------------------------------------------------------
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


## ------------------------------------------------------------------------------------------------------
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


        for i in range(len(self.y)):
            thisx = [0, x1[i], x2[i]]
            thisy = [0, y1[i], y2[i]]

            if i % 30 == 0:
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

            if not self.embedded_fig and i % 30 == 0:
                self.fig.canvas.draw()
                self.fig.canvas.flush_events()






## ------------------------------------------------------------------------------------------------------
## ------------------------------------------------------------------------------------------------------
class DoublePendulumClassic(DoublePendulumRoot):
    """
    This is the classic implementation of Double Pendulum with 7 dimensional state space including derived
    accelerations of both the poles and the input torque. The dynamics of the system are inherited from the Double
    Pendulum Root class.
    """



    C_TYPE = 'Environment'
    C_NAME = 'DoublePendulumClassic'

## -----------------------------------------------------------------------------------------------------
    def __init__(self, p_logging=Log.C_LOG_ALL, t_step=0.2, t_act=5, max_torque=1,
                 max_speed=10, l1=1.0, l2=1.0, m1=1.0, m2=1.0, init_angles='down',
                 g=9.8, history_length=2):

        super().__init__(p_logging, t_step, t_act, max_torque,max_speed, l1, l2, m1, m2, init_angles,
                         g, history_length)


## -----------------------------------------------------------------------------------------------------
    def setup_spaces(self):
        """
        Method to set up the state and action spaces of the classic Double Pendulum Environment. Inheriting from the
        root class, this method adds 3 dimensions for accelerations and torque respectively.
        """
        state_space, action_space = super().setup_spaces()
        state_space.add_dim(
            Dimension(p_name_long='acc 1', p_name_short='a1', p_description='Angular Acceleration of Pendulum 1',
                      p_name_latex='',p_unit='degrees/second^2', p_unit_latex='\text/s^2', p_boundaries=[-6732.31, 5870.988]))

        state_space.add_dim(
            Dimension(p_name_long='acc 2', p_name_short='a2', p_description='Angular Acceleration of Pendulum 2',
                      p_name_latex='',p_unit='degrees/second^2', p_unit_latex='\text/s^2', p_boundaries=[-9650.26, 6805.587]))

        state_space.add_dim(
            Dimension(p_name_long='torque', p_name_short='tau', p_description='input torque',
                      p_name_latex='', p_unit='Newton times meters', p_unit_latex='\tNm',
                      p_boundaries=[-self.max_torque, self.max_torque]))

        return state_space, action_space


## ------------------------------------------------------------------------------------------------------
    def _reset(self, p_seed=None) -> None:
        """
        This method is used to reset the environment.

        Parameters
        ----------
        p_seed : int, optional
            Not yet implemented. The default is None.

        """
        super()._reset()
        self.a1 = 0
        self.a2 = 0
        for i in self._state_space.get_dim_ids()[-2:-4:-1]:
            self._state.set_value(i, 0)


## ------------------------------------------------------------------------------------------------------
    def _simulate_reaction(self, p_state:State, p_action:Action):
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

        den1 = (self.m1 + self.m2) * self.l1 - self.m2 * self.l1 * cos(delta) * cos(delta)
        state[4] = ((self.m2 * self.l1 * state[1] * state[1] * sin(delta) * cos(delta)
                     + self.m2 * self.g * sin(state[2]) * cos(delta)
                     + self.m2 * self.l2 * state[3] * state[3] * sin(delta)
                     - (self.m1 + self.m2) * self.g * sin(state[0]) - torque)
                    / den1)

        den3 = (self.l2 / self.l1) * den1
        state[5] = ((- self.m2 * self.l2 * state[3] * state[3] * sin(delta) * cos(delta)
                     + (self.m1 + self.m2) * self.g * sin(state[0]) * cos(delta)
                     - (self.m1 + self.m2) * self.l1 * state[1] * state[1] * sin(delta)
                     - (self.m1 + self.m2) * self.g * sin(state[2]))
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



## ------------------------------------------------------------------------------------------------------
    def _data_normalization(self, p_state):
        """
        This method is called to normalize any data in between -1 to 1 by considering their boundaries.
        If the boundaries are infinity, then the data is not normalized.

        Parameters
        ----------
        p_value : float
            Input values.
        p_boundaries : Array
            The min-max boundaries of the parameter, e.g. [min, max]

        Returns
        -------
        normalized_value: float

        """
        state = p_state
        for i,j in enumerate(self.get_state_space().get_dim_ids()):
            boundaries = self._state_space.get_dim(j).get_boundaries()
            state[i] = (2 * ((state[i] - min(boundaries))
                             / (max(boundaries) - min(boundaries))) - 1)


        return state


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
                Reward values.
        """

        current_reward = Reward()
        state = p_state_new.get_values().copy()
        p_state_normalized = self._data_normalization(state)
        norm_state = State(self.get_state_space())
        norm_state.set_values(p_state_normalized)
        goal_state = State(self.get_state_space())
        goal_state.set_values([0,0,0,0,0,0,0])

        max_values = []
        min_values = []
        for i in self._state_space.get_dim_ids():
            boundaries = self._state_space.get_dim(i).get_boundaries()
            max_values.append(boundaries[1])
            min_values.append(boundaries[0])


        max_values = self._data_normalization(max_values)
        max_state = State(self.get_state_space())
        max_state.set_values(max_values)
        min_values = self._data_normalization(min_values)
        min_state = State(self.get_state_space())
        min_state.set_values(min_values)

        d_max = ESpace.distance(ESpace, max_state, min_state)
        d = ESpace.distance(ESpace, norm_state, goal_state)
        value = d_max - d
        current_reward.set_overall_reward(value)

        return current_reward




## ------------------------------------------------------------------------------------------------------
## ------------------------------------------------------------------------------------------------------
class DoublePendulumStatic(DoublePendulumRoot):



    C_TYPE = ''
    C_NAME = ''


## ------------------------------------------------------------------------------------------------------
    def __init__(self, p_logging=Log.C_LOG_ALL, t_step=0.2, t_act=5, max_torque=1,
                 max_speed=10, l1=1.0, l2=1.0, m1=1.0, m2=1.0, init_angles='down',
                 g=9.8, history_length=2):
        super().__init__(p_logging=p_logging, t_step=t_step, t_act=t_act, max_torque=max_torque,
                 max_speed=max_speed, l1=l1, l2=l2, m1=m1, m2=m2, init_angles=init_angles,
                 g=g, history_length=history_length)


## ------------------------------------------------------------------------------------------------------
    def _reset(self, p_seed) -> None:
        super()._reset(p_seed)


## ------------------------------------------------------------------------------------------------------
    def _setup_spaces(self, p_state_space: MSpace, p_action_space: MSpace):
        return p_state_space, p_action_space


## ------------------------------------------------------------------------------------------------------
    def _simulate_reaction(self, p_state:State, p_action:Action):
        state = p_state.get_values()
        for i in [0, 2]:
            if state[i] == 0:
                state[i] = 180
            elif state[i] != 0:
                if state[i] > 0:
                    sign = 1
                else:
                    sign = -1
                state[i] = sign * (abs(state[i]) - 180)
        self.th1, self.th1dot, self.a1, self.th2, self.th2dot, self.a2 = state

        torque = p_action.get_sorted_values()[0]
        torque = np.clip(torque, -self.max_torque, self.max_torque)
        torque = tuple(torque.reshape([1]))

        state = np.radians(state)

        state = super()._simulate_reaction(state, torque[0])
        state = np.degrees(state)
        self.action_cw = True if torque[0] > 0 else False
        state_ids = self._state.get_dim_ids()

        for i in [0, 2]:
            if state[i] % 360 < 180:
                state[i] = state[i] % 360
            elif state[i] % 360 > 180:
                state[i] = state[i] % 360 - 360
            if state[i] > 0:
                sign = 1
            else:
                sign = -1
            state[i] = sign * (abs(state[i]) - 180)

        current_state = State(self._state_space)
        for i in range(len(state)):
            current_state.set_value(state_ids[i], state[i])

        return current_state


## -----------------------------------------------------------------------------------------------
    def _compute_reward(self, p_state_old:State, p_state_new:State):
        return super()._compute_reward(p_state_old=p_state_old, p_state_new=p_state_new)