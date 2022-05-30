## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - A Synoptic Framework for Standardized Machine Learning Tasks
## -- Package : mlpro
## -- Module  : howto_rl_006_run_own_agents_with_petting_zoo_environment.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2021-08-26  0.0.0     SY       Creation
## -- 2021-08-27  1.0.0     SY       Released first version
## -- 2021-09-11  1.0.0     MRD      Change Header information to match our new library name
## -- 2021-09-23  1.1.0     SY       Updated wrapper WrEnvPZoo class, provides two different envs
## -- 2021-09-29  1.1.1     SY       Change name: WrEnvPZoo to WrEnvPZOO2MLPro
## -- 2021-10-06  1.1.2     DA       Refactoring 
## -- 2021-10-18  1.1.3     DA       Refactoring 
## -- 2021-11-15  1.1.4     DA       Refactoring 
## -- 2021-11-15  1.1.4     DA       Refactoring 
## -- 2021-11-16  1.1.5     DA       Added explicit scenario reset with constant seeding 
## -- 2021-12-03  1.1.6     DA       Refactoring 
## -- 2022-02-25  1.1.7     SY       Refactoring due to auto generated ID in class Dimension
## -- 2022-05-19  1.1.8     SY       Utilize RandomGenerator
## -- 2022-05-30  1.1.9     DA       Cleaned up/rearranged a bit
## -- 2022-05-30  1.1.8     SY       Update pistonball_v5 to pistonball_v6
## -------------------------------------------------------------------------------------------------

"""
Ver. 1.1.9 (2022-05-30)

This module shows how to run an own policy inside the MLPro standard agent model with a wrapped Petting Zoo environment.
"""


from pettingzoo.butterfly import pistonball_v6
from pettingzoo.classic import connect_four_v3
from mlpro.bf.math import *
from mlpro.rl.models import *
from mlpro.wrappers.pettingzoo import WrEnvPZOO2MLPro
import random
from mlpro.rl.pool.policies.randomgenerator import RandomGenerator



# 1 RL Scenario based on PettingZoo Pistonball environment
class PBScenario (RLScenario):
    """
    Reference : https://www.pettingzoo.ml/butterfly/pistonball
    """

    C_NAME      = 'Pistonball V6'

    def _setup(self, p_mode, p_ada, p_logging):
        zoo_env             = pistonball_v6.env()
        self._env           = WrEnvPZOO2MLPro(zoo_env, p_logging=p_logging)
        
        multi_agent         = MultiAgent(p_name='Pistonball_agents', p_ada=1, p_logging=True)
        agent_idx           = 0
        for k in self._env._zoo_env.action_spaces:
            agent_name      = "Agent_"+str(agent_idx)
            as_ids          = self._env.get_action_space().get_dim_ids()
            agent_ospace    = self._env.get_state_space()
            agent_asspace   = self._env.get_action_space().spawn([as_ids[agent_idx]])
            agent           = Agent(p_policy=RandomGenerator(p_observation_space=agent_ospace,
                                                             p_action_space=agent_asspace,
                                                             p_buffer_size=10,
                                                             p_ada=p_ada,
                                                             p_logging=p_logging
                                                             ),
                                    p_envmodel=None,
                                    p_id=agent_idx,
                                    p_name=agent_name,
                                    p_ada=p_ada,
                                    p_logging=p_logging
                                    )
            multi_agent.add_agent(p_agent=agent)
            agent_idx += 1

        return multi_agent
    
    

# 2 Alternative RL Scenario based on PettingZoo Connect Four environment
class C4Scenario (RLScenario):
    """
    https://www.pettingzoo.ml/classic/connect_four
    """

    C_NAME      = 'Connect Four V3'

    def _setup(self, p_mode, p_ada, p_logging):
        zoo_env             = connect_four_v3.env()
        self._env           = WrEnvPZOO2MLPro(zoo_env, p_logging=True)
        
        multi_agent         = MultiAgent(p_name='Connect4_Agents', p_ada=1, p_logging=True)
        agent_idx           = 0
        for k in self._env._zoo_env.action_spaces:
            agent_name      = "Agent_"+str(agent_idx)
            as_ids          = self._env.get_action_space().get_dim_ids()
            agent_sspace    = self._env.get_state_space()
            agent_asspace   = self._env.get_action_space().spawn([as_ids[agent_idx]])
            agent           = Agent(p_policy=RandomGenerator(p_observation_space=agent_sspace,
                                                             p_action_space=agent_asspace,
                                                             p_buffer_size=10,
                                                             p_ada=p_ada,
                                                             p_logging=p_logging
                                                             ),
                                    p_envmodel=None,
                                    p_id=agent_idx,
                                    p_name=agent_name,
                                    p_ada=p_ada,
                                    p_logging=p_logging
                                    )
            multi_agent.add_agent(p_agent=agent)
            agent_idx += 1

        return multi_agent

  

# 3 Create scenario and run some cycles
if __name__ == "__main__":
    # 3.1 Parameters for demo mode
    logging     = Log.C_LOG_ALL
    visualize   = True
  
else:
    # 3.2 Parameters for internal unit test
    logging     = Log.C_LOG_NOTHING
    visualize   = False
 

# 3.3 Instantiate one of two prepared demo scenarios
myscenario  = PBScenario(
        p_mode=Mode.C_MODE_SIM,
        p_ada=True,
        p_cycle_limit=100,
        p_visualize=visualize,
        p_logging=logging
)

# myscenario  = C4Scenario(
#         p_mode=Mode.C_MODE_SIM,
#         p_ada=True,
#         p_cycle_limit=100,
#         p_visualize=visualize,
#         p_logging=logging
# )


# 3.4 Reset and run the scenario
myscenario.reset(1)
myscenario.run() 