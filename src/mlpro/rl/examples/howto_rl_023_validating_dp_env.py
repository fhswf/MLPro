## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - A Synoptic Framework for Standardized Machine Learning Tasks
## -- Package : mlpro.rl.examples
## -- Module  : howto_rl_022_validating_dp_env.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2022-10-08  0.0.0     LSB      Creation
## -- 2022-10-08  1.0.0     LSB      Relsease of First Version
## -------------------------------------------------------------------------------------------------


"""
Ver. 0.0.0 (2022-10-08)

This module is to be used to validate the Double Pendulum Environment
"""



from mlpro.rl.pool.envs.doublependulum import *
import numpy as np


# Checking for the internal unit test
if __name__ == '__main__':
    p_input = True
else:
    p_input = False

# Taking input from user
if p_input:
    p_torque = float(input('Enter the amount of torque in Nm:'))
    p_cycles = float(input('Enter the amount of cycles to be executed:'))

else:
    p_torque = 0
    p_cycles = 0



## -----------------------------------------------------------------------------------------------
## -----------------------------------------------------------------------------------------------
class ActionGenerator(Policy):
    '''Action Generation based on user input'''


## -----------------------------------------------------------------------------------------------
    def compute_action(self, p_state:State):

        if p_input:
            my_action_values = np.array([p_torque])
        else:
            my_action_values = np.zeros(1)
        return Action(self._id, self._action_space, my_action_values)


## -----------------------------------------------------------------------------------------------
    def _adapt(self, *p_args) -> bool:
        self.log(self.C_LOG_TYPE_W, 'Sorry I am not adapting anything')
        return False




# 1 Implement the random RL scenario
## -----------------------------------------------------------------------------------------------
## -----------------------------------------------------------------------------------------------
class ScenarioDoublePendulumValidation(RLScenario):

    C_NAME      = 'Double Pendulum with Random Actions'


## -----------------------------------------------------------------------------------------------
    def _setup(self, p_mode, p_ada, p_logging):
        # 1.1 Setup environment
        self._env   = DoublePendulumS7(p_init_angles='down', p_max_torque=10, p_logging=p_logging)


        # 1.2 Setup and return random action agent
        policy_random = ActionGenerator(p_observation_space=self._env.get_state_space(),
                                        p_action_space=self._env.get_action_space(),
                                        p_buffer_size=1,
                                        p_ada=1,
                                        p_logging=p_logging)

        return Agent(
            p_policy=policy_random,
            p_envmodel=None,
            p_name='Smith',
            p_ada=p_ada,
            p_logging=p_logging
        )


## -----------------------------------------------------------------------------------------------
    def _run_cycle(self):

        input_cycle_id =

        success, error, adapted = super()._run_cycle()

        return success, error, adapted
        pass


# 2 Create scenario and run the scenario
if __name__ == "__main__":
    # 2.1 Parameters for demo mode
    cycle_limit         = p_cycles
    logging             = Log.C_LOG_ALL
    visualize           = True
    plotting            = True
else:
    # 2.2 Parameters for unittest
    cycle_limit         = 20
    logging             = Log.C_LOG_NOTHING
    visualize           = False
    plotting            = False



# 3 Create your scenario and run some cycles
myscenario  = ScenarioDoublePendulum(
    p_mode=Mode.C_MODE_SIM,
    p_ada=True,
    p_cycle_limit=cycle_limit,
    p_visualize=visualize,
    p_logging=logging
)


myscenario.reset(p_seed=3)
myscenario.run()