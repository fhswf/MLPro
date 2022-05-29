## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - A Synoptic Framework for Standardized Machine Learning Tasks
## -- Package : mlpro
## -- Module  : howto_rl_004_run_multi_agent_with_own_policy_in_multicartpole_environment.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2021-05-20  0.0.0     DA       Creation
## -- 2021-06-06  1.0.0     DA       Release of first version
## -- 2021-08-28  1.1.0     DA       Introduced Policy
## -- 2021-09-11  1.1.0     MRD      Change Header information to match our new library name
## -- 2021 09-26  1.1.1     MRD      Change the import module due to the change of the pool
## --                                folder structer
## -- 2021-10-06  1.1.2     DA       Refactoring 
## -- 2021-10-18  1.1.3     DA       Refactoring 
## -- 2021-11-15  1.2.0     DA       Refactoring 
## -- 2022-02-25  1.2.1     SY       Refactoring due to auto generated ID in class Dimension
## -------------------------------------------------------------------------------------------------

"""
Ver. 1.2.1 (2022-02-25)

This module shows how to run an own multi-agent with the enhanced multi-action environment 
MultiCartPole based on the OpenAI Gym CartPole environment.
"""


from mlpro.rl.models import *
from mlpro.rl.pool.envs.multicartpole import MultiCartPole
import random




# 1 Implement your own agent policy
class MyPolicy (Policy):

    C_NAME      = 'MyPolicy'

    def set_random_seed(self, p_seed=None):
        random.seed(p_seed)


    def compute_action(self, p_state: State) -> Action:
        # 1 Create a numpy array for your action values 
        my_action_values = np.zeros(self._action_space.get_num_dim())

        # 2 Computing action values is up to you...
        for d in range(self._action_space.get_num_dim()):
            my_action_values[d] = random.random() 

        # 3 Return an action object with your values
        return Action(self._id, self._action_space, my_action_values)


    def _adapt(self, *p_args) -> bool:
        # 1 Adapting the internal policy is up to you...
        self.log(self.C_LOG_TYPE_W, 'Sorry, I am a stupid agent...')

        # 2 Only return True if something has been adapted...
        return False




# 2 Implement your own RL scenario
class MyScenario (RLScenario):

    C_NAME      = 'Matrix'

    def _setup(self, p_mode, p_ada, p_logging):

        # 1 Setup Multi-Agent Environment (consisting of 3 OpenAI Gym Cartpole envs)
        self._env   = MultiCartPole(p_num_envs=3, p_logging=p_logging)


        # 2 Setup Multi-Agent 

        # 2.1 Create empty Multi-Agent
        multi_agent = MultiAgent(
            p_name='Smith',
            p_ada=True,
            p_logging=p_logging
        )

        # 2.2 Add Single-Agent #1 with own policy (controlling sub-environment #1)
        ss_ids = self._env.get_state_space().get_dim_ids()
        as_ids = self._env.get_action_space().get_dim_ids()
        multi_agent.add_agent(
            p_agent=Agent(
                p_policy=MyPolicy(
                    p_observation_space=self._env.get_state_space().spawn([ss_ids[0],ss_ids[1],ss_ids[2],ss_ids[3]]),
                    p_action_space=self._env.get_action_space().spawn([as_ids[0]]),
                    p_buffer_size=1,
                    p_ada=True,
                    p_logging=p_logging
                ),
                p_envmodel=None,
                p_name='Smith-1',
                p_id=0,
                p_ada=True,
                p_logging=p_logging
            ),
            p_weight=0.3
        )

        # 2.3 Add Single-Agent #2 with own policy (controlling sub-environments #2,#3)
        multi_agent.add_agent(
            p_agent=Agent(
                p_policy=MyPolicy(
                    p_observation_space=self._env.get_state_space().spawn([ss_ids[4],ss_ids[5],ss_ids[6],ss_ids[7],ss_ids[8],ss_ids[9],ss_ids[10],ss_ids[11]]),
                    p_action_space=self._env.get_action_space().spawn([as_ids[1],as_ids[2]]),
                    p_buffer_size=1,
                    p_ada=True,
                    p_logging=p_logging
                ),
                p_envmodel=None,
                p_name='Smith-2',
                p_id=1,
                p_ada=True,
                p_logging=p_logging
            ),
            p_weight=0.7
        )

        # 2.4 Adaptive ML model (here: our multi-agent) is returned
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
 

myscenario  = MyScenario(
        p_mode=Mode.C_MODE_SIM,
        p_ada=True,
        p_cycle_limit=200,
        p_visualize=visualize,
        p_logging=logging
)

myscenario.run() 