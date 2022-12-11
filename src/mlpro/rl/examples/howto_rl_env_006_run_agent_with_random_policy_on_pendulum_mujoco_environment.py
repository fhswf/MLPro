## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - A Synoptic Framework for Standardized Machine Learning Tasks
## -- Package : mlpro
## -- Module  : howto_rl_env_006_run_agent_with_random_policy_on_pendulum_mujoco_environment.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2022-09-17  0.0.0     MRD       Creation
## -- 2022-12-11  0.0.1     MRD       Refactor due to new bf.Systems
## -- 2022-12-11  1.0.0     MRD       First Release
## -------------------------------------------------------------------------------------------------


"""
Ver. 1.0.0 (2022-12-11)

This module shows how to run a random policy on Pendulum with MuJoCo Simulation.
"""


import random
import numpy as np

from mlpro.bf.ml import Model
from mlpro.bf.ops import Mode
from mlpro.bf.various import Log
from mlpro.rl.models_agents import Policy, Agent
from mlpro.rl.models_train import RLScenario
from mlpro.bf.systems import State, Action
from mlpro.rl.models_env_ada import SARSElement
from mlpro.rl.pool.envs.mujoco.pendulum import Pendulum

# 1 Implement your own agent policy
class MyPolicy (Policy):

    C_NAME      = 'MyPolicy'

    def set_random_seed(self, p_seed=None):
        random.seed(p_seed)


    def compute_action(self, p_state: State) -> Action:
        # 1.1 Create a numpy array for your action values 
        my_action_values = np.zeros(self._action_space.get_num_dim())

        # 1.2 Computing action values is up to you...
        for d in range(self._action_space.get_num_dim()):
            my_action_values[d] = np.random.uniform(-50, 50)

        # 1.3 Return an action object with your values
        return Action(self._id, self._action_space, my_action_values)


    def _adapt(self, p_sars_elem: SARSElement) -> bool:
        # 1.4 Adapting the internal policy is up to you...
        self.log(self.C_LOG_TYPE_W, 'Sorry, I am a stupid agent...')

        # 1.5 Only return True if something has been adapted...
        return False


# 2 Implement your own RL scenario
class MyScenario (RLScenario):

    C_NAME      = 'Matrix'

    def _setup(self, p_mode, p_ada: bool, p_visualize:bool, p_logging) -> Model:
        # 2.1 Setup environment
        self._env   = Pendulum() 

        # 2.2 Setup standard single-agent with own policy
        return Agent( p_policy=MyPolicy( p_observation_space=self._env.get_state_space(),
                                         p_action_space=self._env.get_action_space(),
                                         p_buffer_size=1,
                                         p_ada=p_ada,
                                         p_visualize=p_visualize,
                                         p_logging=p_logging),    
                      p_envmodel=None,
                      p_name='Smith',
                      p_ada=p_ada,
                      p_visualize=p_visualize,
                      p_logging=p_logging)

# 3 Create scenario and run some cycles
if __name__ == "__main__":
    # 3.1 Parameters for demo mode
    cycle_limit = 2000
    logging     = Log.C_LOG_ALL
    visualize   = False
  
else:
    # 3.2 Parameters for internal unit test
    cycle_limit = 10
    logging     = Log.C_LOG_NOTHING
    visualize   = False
 

# 3.3 Create your scenario and run some cycles
myscenario  = MyScenario(
        p_mode=Mode.C_MODE_SIM,
        p_ada=True,
        p_cycle_limit=cycle_limit,
        p_visualize=visualize,
        p_logging=logging
)

myscenario.reset(p_seed=3)

myscenario.run()