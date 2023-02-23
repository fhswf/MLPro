## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - A Synoptic Framework for Standardized Machine Learning Tasks
## -- Package : mlpro.bf.systems.pool
## -- Module  : doublependulum.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2023-mm-dd  0.0.0     LSB      Creation
## -------------------------------------------------------------------------------------------------



from mlpro.rl.pool.envs.doublependulum import *


## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class DPFctSTrans(FctSTrans):

    def _simulate_reaction(self, p_state: State, p_action: Action) -> State:
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

    def _derivs(self, p_state, t, p_torque):
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
                    - (self._m1 + self._m2) * self._g * sin(p_state[0]) - p_torque)
                   / den1)

        dydx[2] = p_state[3]

        den2 = (self._l2 / self._l1) * den1
        dydx[3] = ((- self._m2 * self._l2 * p_state[3] * p_state[3] * sin(delta) * cos(delta)
                    + (self._m1 + self._m2) * self._g * sin(p_state[0]) * cos(delta)
                    - (self._m1 + self._m2) * self._l1 * p_state[1] * p_state[1] * sin(delta)
                    - (self._m1 + self._m2) * self._g * sin(p_state[2]))
                   / den2)

        return dydx



## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class DPFctBroken(FctBroken):

    def _compute_broken(self, p_state: State) -> bool:
        return False


## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class DPFctSuccess(FctSuccess):

    def _compute_success(self, p_state: State) -> bool:
        return False


## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class DPSystem(DoublePendulumRoot, System):


## -------------------------------------------------------------------------------------------------
    def __init__(self,
                   p_mode = Mode.C_MODE_SIM,
                   p_fct_strans : FctSTrans = DPFctSTrans(),
                   p_fct_success : FctSuccess = DPFctSuccess(),
                   p_fct_broken : FctBroken = DPFctBroken(),
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
                   p_plot_level:int=2,
                   p_rst_balancing = DoublePendulumRoot.C_RST_BALANCING_002,
                   p_rst_swinging = DoublePendulumRoot.C_RST_SWINGING_001,
                   p_rst_swinging_outer_pole = DoublePendulumRoot.C_RST_SWINGING_OUTER_POLE_001,
                   p_reward_trend: bool = False,
                   p_reward_window:int = 0,
                   p_random_range:list = None,
                   p_balancing_range:list = (-0.2,0.2),
                   p_swinging_outer_pole_range = (0.2,0.5),
                   p_break_swinging:bool = False,
                   p_logging=Log.C_LOG_ALL):

        DoublePendulumRoot.__init__(self,
                                    p_mode=p_mode,
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
                                    p_plot_level=p_plot_level,
                                    p_rst_balancing = p_rst_balancing,
                                    p_rst_swinging = p_rst_swinging,
                                    p_rst_swinging_outer_pole = p_rst_swinging_outer_pole,
                                    p_reward_trend=p_reward_trend,
                                    p_reward_window=p_reward_window,
                                    p_random_range=p_random_range,
                                    p_balancing_range=p_balancing_range,
                                    p_swinging_outer_pole_range = p_swinging_outer_pole_range,
                                    p_break_swinging=p_break_swinging,
                                    p_logging = p_logging)

        System.__init__(self,
                        p_mode=p_mode,
                        p_latency=p_latency,
                        p_fct_strans=p_fct_strans,
                        p_fct_success=p_fct_success,
                        p_fct_broken=p_fct_broken,
                        p_logging=p_logging)


        