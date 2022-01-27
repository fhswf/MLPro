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
## -------------------------------------------------------------------------------------------------

"""
Ver. 0.9.0 (2022-01-26)

This module provides an RL environment of double pendulum.
"""


from mlpro.rl.models import *
from mlpro.bf.various import *
import numpy as np
from numpy import sin, cos
import random
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import scipy.integrate as integrate
from collections import deque



## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class DoublePendulum (Environment):
    """
    This is the main class of the Double Pendulum environment that inherits 
    Environment class from MLPro.
    
    Parameters
    ----------
    p_logging : Log, optional
        logging functionalities. The default is Log.C_LOG_ALL.
    t_step : float, optional
        time for each time step (in seconds). The default is 0.01.
    t_act : int, optional
        action frequency (with respect to the time step). The default is 20.
    max_torque : float, optional
        maximum torque applied to pendulum 1. The default is 20.
    max_speed : float, optional
        maximum speed applied to pendulum 1. The default is 10.
    l1 : float, optional
        length of pendulum 1 in m. The default is 0.5
    l2 : float, optional
        length of pendulum 2 in m. The default is 0.5
    m1 : float, optional
        mass of pendulum 1 in kg. The default is 0.5
    m2 : float, optional
        mass of pendulum 2 in kg. The default is 0.5
    th1 : float, optional 
        initial angle of pendulum 1 in degrees. The default is 0.0
    th1dot : float, optional 
        initial angular velocities of pendulum 1 in degrees per second. The default is 0.0
    th2 : float, optional 
        initial angle of pendulum 2 in degrees. The default is 0.0
    th2dot : float, optional 
        initial angular velocities of pendulum 2 in degrees per second. The default is 0.0
    g : float, optional
        gravitational acceleration. The default is 9.8
    history_length : int, optional
        historical trajectory points to display. The default is 20.
        
    Attributes
    ----------
    C_NAME : str
        name of the environment.
    C_CYCLE_LIMIT : int
        the number of cycle limit.
    C_LATENCY : timedelta()
        latency.
    C_REWARD_TYPE : Reward
        rewarding type.
    """
    C_NAME              = "DoublePendulum"
    C_CYCLE_LIMIT       = 0
    C_LATENCY           = timedelta(0,0,0)    
    C_REWARD_TYPE       = Reward.C_TYPE_OVERALL
    
    
## -------------------------------------------------------------------------------------------------
    def __init__(self, p_logging=Log.C_LOG_ALL, t_step=0.01, t_act=20, max_torque=20,
                max_speed=10, l1=0.5, l2=0.5, m1=0.5, m2=0.5, th1=0.0, th2=0.0, 
                th1dot=0.0, th2dot=0.0, g=9.8, history_length=20):
        self.t_step = t_step
        self.t_act = t_act
        self.set_latency(timedelta(0,t_act*t_step,0))
        
        self.max_torque = max_torque
        self.max_speed = max_speed
        self.l1 = l1
        self.l2 = l2
        self.L = l1+l2
        self.m1 = m1
        self.m2 = m2
        self.th1 = th1
        self.th2 = th2
        self.th1dot = th1dot
        self.th2dot = th2dot
        self.g = g
        
        self.history_x = deque(maxlen=history_length)
        self.history_y = deque(maxlen=history_length)
        
        super().__init__(p_mode=Environment.C_MODE_SIM, p_logging=p_logging)
        self._state = State(self._state_space)
        
        self.reset()  


## -------------------------------------------------------------------------------------------------
    def setup_spaces(self):
        """
        This method is used to setup action and state spaces of the system.

        Returns
        -------
        state_space : ESpace()
            state space of the system.
        action_space : ESpace()
            action space of the system.

        """
        state_space     = ESpace()
        action_space    = ESpace()
        
        state_space.add_dim(Dimension(0, 'theta 1', 'th1', 'Angle of Pendulum 1', '', 'degrees', 
                            '\textdegrees',[-np.pi, np.pi]))
        state_space.add_dim(Dimension(1, 'omega 1', 'w1', 'Angular Velocity of Pendulum 1', '', 
                            'degrees/second', '\textdegrees/s',[-np.inf, np.inf]))
        state_space.add_dim(Dimension(2, 'theta 2', 'th2', 'Angle of pendulum 2', '', 'degrees', 
                            '\textdegrees',[-np.pi, np.pi]))
        state_space.add_dim(Dimension(3, 'omega 2', 'w2', 'Angular Velocity of Pendulum 2', '', 
                            'degrees/second', '\textdegrees/s',[-np.inf, np.inf]))
        
        action_space.add_dim(Dimension(0, 'torque 1', 'tau1', 'Applied Torque of Motor 1', '', 
                            'Nm', 'Nm', [-self.max_torque, self.max_torque]))

        return state_space, action_space
        

## -------------------------------------------------------------------------------------------------
    def derivs(self, state, t):
        dydx = np.zeros_like(state)
        dydx[0] = state[1]

        delta = state[2] - state[0]
        den1 = (self.m1+self.m2) * self.l1 - self.m2 * self.l1 * cos(delta) * cos(delta)
        dydx[1] = ((self.m2 * self.l1 * state[1] * state[1] * sin(delta) * cos(delta)
                    + self.m2 * self.g * sin(state[2]) * cos(delta)
                    + self.m2 * self.l2 * state[3] * state[3] * sin(delta)
                    - (self.m1+self.m2) * self.g * sin(state[0]))
                   / den1)

        dydx[2] = state[3]

        den2 = (self.l2/self.l1) * den1
        dydx[3] = ((- self.m2 * self.l2 * state[3] * state[3] * sin(delta) * cos(delta)
                    + (self.m1+self.m2) * self.g * sin(state[0]) * cos(delta)
                    - (self.m1+self.m2) * self.l1 * state[1] * state[1] * sin(delta)
                    - (self.m1+self.m2) * self.g * sin(state[2]))
                   / den2)

        return dydx
        

## -------------------------------------------------------------------------------------------------
    @staticmethod
    def angle_normalize(x):
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
        self._state.set_value(0, np.radians(self.th1))
        self._state.set_value(1, np.radians(self.th1dot))
        self._state.set_value(2, np.radians(self.th2))
        self._state.set_value(3, np.radians(self.th2dot))
        
        self.history_x.clear()
        self.history_y.clear()


## -------------------------------------------------------------------------------------------------
    def _simulate_reaction(self, p_state:State, p_action:Action) -> State:
        """
        This method is used to calculate the next states of the system after a set of actions.

        Parameters
        ----------
        p_state : State
            State.
        p_action : Action
            ACtion.

        Returns
        -------
        _state : State
            current states.

        """
        state = p_state.get_values()
        th1, th1dot, th2, th2dot = state
        torque = p_action.get_sorted_values()[0]
        torque = np.clip(torque, -self.max_torque, self.max_torque)
        
        state[1] = th1dot + (3 * self.g / (2 * self.l1) * sin(th1) + 3.0 / 
                (self.m1 * self.l1 ** 2) * torque) * self.t_step
                
        if abs(th1dot) > self.max_speed:
            state[1] = np.clip(state[1], -th1dot, th1dot)
        else:
            state[1] = np.clip(state[1], -self.max_speed, self.max_speed)
        
        self.y = integrate.odeint(self.derivs, state, np.arange(0, self.t_act*self.t_step, self.t_step))
        state = self.y[-1]
        
        for i in range(len(state)):
            self._state.set_value(i, state[i])

        return self._state


## -------------------------------------------------------------------------------------------------
    def _compute_broken(self, p_state:State) -> bool:
        """
        This method computes the broken flag. This method can be redefined.

        Parameters
        ----------
        p_state : State
            state.

        Returns
        -------
        bool
            broken or not.

        """ 

        return False


## -------------------------------------------------------------------------------------------------
    def _compute_success(self, p_state:State) -> bool:
        """
        This method computes the success flag. This method can be redefined.

        Parameters
        ----------
        p_state : State
            state.

        Returns
        -------
        bool
            success or not success.

        """
        
        return False
    

## -------------------------------------------------------------------------------------------------
    def _compute_reward(self, p_state_old:State, p_state_new:State) -> Reward:
        reward = Reward(Reward.C_TYPE_OVERALL)
        count = 0
        for th1 in self.y[:,0]:
            if np.degrees(th1) > 179 or np.degrees(th1) < 181 or \
                np.degrees(th1) < -179 or np.degrees(th1) > -181:
                count += 1
        
        reward.set_overall_reward(count/len(self.y))
        
        return reward
    

## -------------------------------------------------------------------------------------------------
    def init_plot(self, p_figure=None):
        if hasattr(self, 'fig'):
            plt.close(self.fig)
        self.fig = plt.figure(figsize=(5,4)) if p_figure==None else p_figure
        self.ax = self.fig.add_subplot(autoscale_on=False, 
                    xlim=(-self.L*1.2, self.L*1.2), ylim=(-self.L*1.2, self.L*1.2))
        self.ax.set_aspect('equal')
        self.ax.grid()
        
        self.line, = self.ax.plot([], [], 'o-', lw=2)
        self.trace, = self.ax.plot([], [], '.-', lw=1, ms=2)


## -------------------------------------------------------------------------------------------------
    def update_plot(self):
        x1 = self.l1*sin(self.y[:, 0])
        y1 = -self.l1*cos(self.y[:, 0])

        x2 = self.l2*sin(self.y[:, 2]) + x1
        y2 = -self.l2*cos(self.y[:, 2]) + y1
        
        # def animate(i):
            # thisx = [0, x1[i], x2[i]]
            # thisy = [0, y1[i], y2[i]]

            # self.history_x.appendleft(thisx[2])
            # self.history_y.appendleft(thisy[2])

            # self.line.set_data(thisx, thisy)
            # self.trace.set_data(self.history_x, self.history_y)
            # return self.line, self.trace

        # ani = animation.FuncAnimation(
            # self.fig, animate, len(self.y), interval=self.t_step*1000, blit=True, repeat=False)

        thisx = [0, x1[-1], x2[-1]]
        thisy = [0, y1[-1], y2[-1]]            
        self.history_x.appendleft(thisx[2])
        self.history_y.appendleft(thisy[2])
        self.line.set_data(thisx, thisy)
        self.trace.set_data(self.history_x, self.history_y)
        
        plt.show()