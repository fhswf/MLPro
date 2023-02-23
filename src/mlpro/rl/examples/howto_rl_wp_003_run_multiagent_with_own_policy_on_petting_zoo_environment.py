## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - A Synoptic Framework for Standardized Machine Learning Tasks
## -- Package : mlpro.rl.examples
## -- Module  : howto_rl_wp_003_run_multiagent_with_own_policy_on_petting_zoo_environment.py
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
## -- 2022-10-08  1.2.0     SY       Turn off render: causing error due to pzoo ver 1.22.0 
## -- 2022-10-14  1.2.1     SY       Refactoring 
## -- 2022-11-01  1.2.2     DA       Refactoring 
## -- 2022-11-02  1.2.3     SY       Unable logging in unit test model and bug fixing
## -- 2022-11-07  1.3.0     DA       Refactoring
## -- 2023-02-21  1.4.0     DA       Added save + reload + rerun steps to demonstrate/validate
## --                                persistence of pettingzoo scenarios
## -------------------------------------------------------------------------------------------------

"""
Ver. 1.4.0 (2023-02-21)

This module shows how to run an own policy inside the MLPro standard agent model with a wrapped
Petting Zoo environment.

You will learn:
    
1) How to set up a scenario for a Petting Zoo environment in MLPro

2) How to run the scenario

3) How to save, reload and rerun a scenario
    
"""


from pathlib import Path
from pettingzoo.butterfly import pistonball_v6
from pettingzoo.classic import connect_four_v3
from mlpro.bf.math import *
from mlpro.rl import *
from mlpro.wrappers.pettingzoo import WrEnvPZOO2MLPro
from mlpro.rl.pool.policies.randomgenerator import RandomGenerator





# 1 RL Scenario based on PettingZoo Pistonball environment
class PBScenario (RLScenario):
    """
    Reference : https://www.pettingzoo.ml/butterfly/pistonball
    """

    C_NAME      = 'Pistonball V6'

    def _setup(self, p_mode, p_ada: bool, p_visualize: bool, p_logging) -> Model:
        if p_visualize:
            zoo_env         = pistonball_v6.env(render_mode="human")
        else:
            zoo_env         = pistonball_v6.env(render_mode="ansi")
        self._env           = WrEnvPZOO2MLPro(zoo_env, p_visualize=p_visualize, p_logging=p_logging)
        
        multi_agent         = MultiAgent(p_name='Pistonball_agents', p_ada=1, p_visualize=p_visualize, p_logging=p_logging)
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
                                                             p_visualize=p_visualize,
                                                             p_logging=p_logging
                                                             ),
                                    p_envmodel=None,
                                    p_id=agent_idx,
                                    p_name=agent_name,
                                    p_ada=p_ada,
                                    p_visualize=p_visualize,
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

    def _setup(self, p_mode, p_ada: bool, p_visualize: bool, p_logging) -> Model:
        if p_visualize:
            zoo_env         = connect_four_v3.env(render_mode="human")
        else:
            zoo_env         = connect_four_v3.env(render_mode="ansi")
        self._env           = WrEnvPZOO2MLPro(zoo_env, p_visualize=p_visualize, p_logging=p_logging)
        
        multi_agent         = MultiAgent(p_name='Connect4_Agents', p_ada=1, p_visualize=p_visualize, p_logging=p_logging)
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
                                                             p_visualize=p_visualize, 
                                                             p_logging=p_logging
                                                             ),
                                    p_envmodel=None,
                                    p_id=agent_idx,
                                    p_name=agent_name,
                                    p_ada=p_ada,
                                    p_visualize=p_visualize, 
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
    now         = datetime.now()
    path        = str(Path.home()) + os.sep + '%04d-%02d-%02d  %02d.%02d.%02d Howto RL WP 003' % (now.year, now.month, now.day, now.hour, now.minute, now.second)
else:
    # 3.2 Parameters for internal unit test
    logging     = Log.C_LOG_NOTHING
    visualize   = False
    path        = None
 

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


if __name__ == '__main__':
    # 3.5 In demo mode we save, reload and rerun the entire scenario to demonstrate persistence
    myscenario.save(path, 'dummy')
    input('\nPress ENTER to reload and run again...\n')
    myscenario = PBScenario.load(path + os.sep + 'scenario')
    myscenario.reset(1)
    myscenario.run() 

