## -------------------------------------------------------------------------------------------------
## -- Project : FH-SWF Automation Technology - Common Code Base (CCB)
## -- Package : mlpro
## -- Module  : Howto 08 - (RL) Run own agents with petting zoo environment
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2021-08-26  0.0.0     SY       Creation
## -- 2021-08-27  1.0.0     SY       Released first version
## -- 2021-09-11  1.0.0     MRD      Change Header information to match our new library name
## -- 2021-09-23  1.1.0     SY       Updated wrapper WrEnvPZoo class, provides two different envs
## -- 2021-09-29  1.1.1     SY       Change name: WrEnvPZoo to WrEnvPZOO2MLPro
## -- 2021-10-06  1.1.2     DA       Refactoring 
## -------------------------------------------------------------------------------------------------

"""
Ver. 1.1.2 (2021-10-06)

This module shows how to run an own policy inside the standard agent model with a Petting Zoo environment using 
the fhswf_at_ml framework.
"""


from pettingzoo.butterfly import pistonball_v4
from pettingzoo.classic import connect_four_v3
from mlpro.bf.math import *
from mlpro.rl.models import *
from mlpro.rl.wrappers import WrEnvPZOO2MLPro
import random


# Piston Ball Scenario
"""
Reference : https://www.pettingzoo.ml/butterfly/pistonball
"""
class ContRandPolicy(Policy):

    C_NAME      = 'ContRandPolicy'

    def compute_action(self, p_state: State) -> Action:
        my_action_values = np.zeros(self._action_space.get_num_dim())
        for d in range(self._action_space.get_num_dim()):
            my_action_values[d] = random.uniform(-1,1) 
        return Action(self._id, self._action_space, my_action_values)


    def _adapt(self, *p_args) -> bool:
        self.log(self.C_LOG_TYPE_W, 'Sorry, I am a stupid agent...')
        return False
    
    
class PBScenario(Scenario):

    C_NAME      = 'Pistonball V4'

    def _setup(self, p_mode, p_ada, p_logging):
        zoo_env             = pistonball_v4.env()
        self._env           = WrEnvPZOO2MLPro(zoo_env, p_logging=True)
        
        self._agent         = MultiAgent(p_name='Pistonball_agents', p_ada=1, p_logging=True)
        agent_id            = 0
        for k in self._env._zoo_env.action_spaces:
            agent_name      = "Agent_"+str(agent_id)
            agent_sspace    = self._env.get_state_space()
            agent_asspace   = self._env.get_action_space().spawn([agent_id])
            agent           = Agent(p_policy=ContRandPolicy(p_state_space=agent_sspace,
                                                            p_action_space=agent_asspace,
                                                            p_buffer_size=10,
                                                            p_ada=p_ada,
                                                            p_logging=p_logging
                                                            ),
                                    p_envmodel=None,
                                    p_id=agent_id,
                                    p_name=agent_name,
                                    p_ada=p_ada,
                                    p_logging=p_logging
                                    )
            self._agent.add_agent(p_agent=agent)
            agent_id += 1
    
    
# Connect Four Scenario
"""
https://www.pettingzoo.ml/classic/connect_four
"""
class DiscRandPolicy(Policy):

    C_NAME      = 'DiscRandPolicy'

    def compute_action(self, p_state: State) -> Action:
        my_action_values = np.zeros(self._action_space.get_num_dim())
        for d in range(self._action_space.get_num_dim()):
            my_action_values[d] = random.randint(0,6) 
        return Action(self._id, self._action_space, my_action_values)


    def adapt(self, *p_args) -> bool:
        if not super().adapt(p_args): return False
        self.log(self.C_LOG_TYPE_W, 'Sorry, I am a stupid agent...')
        return False
    
class C4Scenario(Scenario):

    C_NAME      = 'Connect Four V3'

    def _setup(self, p_mode, p_ada, p_logging):
        zoo_env             = connect_four_v3.env()
        self._env           = WrEnvPZOO2MLPro(zoo_env, p_logging=True)
        
        self._agent         = MultiAgent(p_name='Connect4_Agents', p_ada=1, p_logging=True)
        agent_id            = 0
        for k in self._env._zoo_env.action_spaces:
            agent_name      = "Agent_"+str(agent_id)
            agent_sspace    = self._env.get_state_space()
            agent_asspace   = self._env.get_action_space().spawn([agent_id])
            agent           = Agent(p_policy=DiscRandPolicy(p_state_space=agent_sspace,
                                                            p_action_space=agent_asspace,
                                                            p_buffer_size=10,
                                                            p_ada=p_ada,
                                                            p_logging=p_logging
                                                            ),
                                    p_envmodel=None,
                                    p_id=agent_id,
                                    p_name=agent_name,
                                    p_ada=p_ada,
                                    p_logging=p_logging
                                    )
            self._agent.add_agent(p_agent=agent)
            agent_id += 1



# 3 Instantiate scenario

myscenario  = PBScenario(
    p_mode=Environment.C_MODE_SIM,
    p_ada=True,
    p_cycle_limit=100,
    p_visualize=False,
    p_logging=True
)

# myscenario  = C4Scenario(
#     p_mode=Environment.C_MODE_SIM,
#     p_ada=True,
#     p_cycle_limit=100,
#     p_visualize=False,
#     p_logging=True
# )



# 4 Run some cycles
myscenario.run()
