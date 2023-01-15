## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - A Synoptic Framework for Standardized Machine Learning Tasks
## -- Package : mlpro.rl.examples
## -- Module  : howto_rl_oarl_999_adaptive_environment.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2023-01-12  0.0.0     LSB      Creation
## -------------------------------------------------------------------------------------------------

"""
Ver. 0.0.0 (2023-01-12)
...

You will learn:

1.

2.

3.

"""

from mlpro.rl.pool.envs.doublependulum import DoublePendulumS4
from mlpro.rl.models_env_adaptive_environment import *
from mlpro.rl.models import *
from mlpro.rl.pool.policies.randomgenerator import RandomGenerator
from numpy import integrate






## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
# 1 Implement the random RL scenario
class ScenarioDoublePendulum(RLScenario):

    C_NAME      = 'Double Pendulum with Random Actions'

    def _setup(self, p_mode, p_ada: bool, p_visualize: bool, p_logging) -> Model:
        # 1.1 Setup environment
        self._env   = DoublePendulumS4(p_init_angles='random', p_max_torque=10, p_visualize=p_visualize,
            p_logging=p_logging)


        # 1.2 Setup and return random action agent
        policy_random = RandomGenerator(p_observation_space=self._env.get_state_space(),
                                        p_action_space=self._env.get_action_space(),
                                        p_buffer_size=1,
                                        p_ada=1,
                                        p_visualize=p_visualize,
                                        p_logging=p_logging)

        return Agent(
            p_policy=policy_random,
            p_envmodel=None,
            p_name='Smith',
            p_ada=p_ada,
            p_visualize=p_visualize,
            p_logging=p_logging
        )


class DPFctSTrans(OAFctSTrans):

    def __init__(self,
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
                 p_logging=Log.C_LOG_ALL ):

        OAFctSTrans.__init__(self,
        )

    def _simulate_reaction(self, p_state: State, p_action: Action) -> State:

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
        # torque = np.clip(torque, -self._max_torque, self._max_torque)

        state = np.radians(state)

        # if self._max_torque != 0:
        #     self._alpha = abs(torque) / self._max_torque
        # else:
        #     self._alpha = 0

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

        current_state.set_values(state)

        return current_state


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