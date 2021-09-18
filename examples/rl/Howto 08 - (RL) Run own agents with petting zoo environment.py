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
## -------------------------------------------------------------------------------------------------

"""
Ver. 1.1.0 (2021-08-27)

This module shows how to run an own policy inside the standard agent model with a Petting Zoo environment using 
the fhswf_at_ml framework.


NOT READY TO USE!
"""


from pettingzoo.classic import connect_four_v3
from pettingzoo.butterfly import pistonball_v4
from mlpro.bf.math import *
from mlpro.rl.models import *
from mlpro.rl.wrappers import WrEnvPZoo
import random


class MyPolicy(Policy):

    C_NAME      = 'MyPolicy'

    def compute_action(self, p_state: State) -> Action:
        my_action_values = np.zeros(self._action_space.get_num_dim())
        for d in range(self._action_space.get_num_dim()):
            my_action_values[d] = random.random() 
        return Action(self._id, self._action_space, my_action_values)


    def adapt(self, *p_args) -> bool:
        if not super().adapt(p_args): return False
        self.log(self.C_LOG_TYPE_W, 'Sorry, I am a stupid agent...')
        return False
    
    
class MyScenario(Scenario):

    C_NAME      = 'Pistonball V4'

    def _setup(self, p_mode, p_ada, p_logging):
        # zoo_env             = connect_four_v3.env()
        zoo_env             = pistonball_v4.env()
        self._env           = WrEnvPZoo(zoo_env, p_logging=True)
        
        self._agent         = MultiAgent(p_name='Pistonball_agents', p_ada=1, p_logging=False)
        agent_id            = 1
        for k in self._env._zoo_env.action_spaces:
            agent_name      = "Agent_"+str(agent_id)
            agent_sspace    = self._env.get_state_space().spawn([agent_id-1])
            agent_asspace   = self._env.get_action_space().spawn([agent_id-1])
            agent           = Agent(p_policy=MyPolicy(p_state_space=agent_sspace,
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
myscenario  = MyScenario(
    p_mode=Environment.C_MODE_SIM,
    p_ada=True,
    p_cycle_limit=100,
    p_visualize=False,
    p_logging=True
)



# 4 Run some cycles
myscenario.run()
