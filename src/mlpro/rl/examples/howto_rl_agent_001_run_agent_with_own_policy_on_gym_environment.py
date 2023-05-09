## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - A Synoptic Framework for Standardized Machine Learning Tasks
## -- Package : mlpro.rl.examples
## -- Module  : howto_rl_agent_001_run_agent_with_own_policy_on_gym_environment.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2021-05-09  0.0.0     DA       Creation
## -- 2021-06-06  1.0.0     DA       Released first version
## -- 2021-08-28  1.1.0     DA       Introduced Policy
## -- 2021-09-11  1.1.0     MRD      Change Header information to match our new library name
## -- 2021-09-29  1.1.1     SY       Change name: WrEnvGym to WrEnvGYM2MLPro
## -- 2021-10-06  1.1.2     DA       Refactoring 
## -- 2021-10-18  1.1.3     DA       Refactoring 
## -- 2021-11-15  1.2.0     DA       Refactoring 
## -- 2021-11-16  1.2.1     DA       Added explicit scenario reset with constant seeding 
## -- 2021-12-03  1.2.2     DA       Refactoring 
## -- 2022-07-20  1.2.3     SY       Update due to the latest introduction of Gym 0.25
## -- 2022-10-13  1.2.4     SY       Refactoring 
## -- 2022-11-01  1.2.5     DA       Refactoring 
## -- 2022-11-02  1.2.6     DA       Refactoring 
## -- 2022-11-07  1.3.0     DA       Refactoring 
## -- 2023-01-14  1.3.1     MRD      Removing default parameter new_step_api and render_mode for gym
## -- 2023-04-19  1.3.2     MRD      Refactor module import gym to gymnasium
## -------------------------------------------------------------------------------------------------

"""
Ver. 1.3.2 (2023-04-19)

This module shows how to run an own policy inside the standard agent model with an OpenAI Gym environment using 
MLPro framework.

You will learn:
    
1) How to set up a native policy for an agent
    
2) How to set up an agent
    
3) How to set up a scenario
    
4) How to wrap Gym environment to MLPro environment

5) How to run the scenario
    
"""


from mlpro.bf.math import *
from mlpro.rl import *
from mlpro.wrappers.gymnasium import WrEnvGYM2MLPro
import gymnasium as gym
import random




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
            my_action_values[d] = random.random() 

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
        if p_visualize:
            gym_env     = gym.make('CartPole-v1', render_mode="human")
        else:
            gym_env     = gym.make('CartPole-v1')
        self._env   = WrEnvGYM2MLPro( p_gym_env=gym_env, p_visualize=p_visualize, p_logging=p_logging) 

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
    cycle_limit = 100
    logging     = Log.C_LOG_ALL
    visualize   = True
  
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