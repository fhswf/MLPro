## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - A Synoptic Framework for Standardized Machine Learning Tasks
## -- Package : mlpro.rl.examples
## -- Module  : howto_rl_agent_002_train_agent_with_own_policy_on_gym_environment.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2021-06-03  0.0.0     DA       Creation
## -- 2021-06-06  1.0.0     DA       Released first version
## -- 2021-06-25  1.1.0     DA       Extended by data logging/storing (user home directory)
## -- 2021-07-06  1.1.1     SY       Bugfix due to method Training.save_data() update
## -- 2021-08-28  1.2.0     DA       Introduced Policy
## -- 2021-09-11  1.2.0     MRD      Change Header information to match our new library name
## -- 2021-09-28  1.2.1     SY       Updated due to implementation of method get_cycle_limits()
## -- 2021-09-29  1.2.2     SY       Change name: WrEnvGym to WrEnvGYM2MLPro
## -- 2021-10-06  1.2.3     DA       Refactoring 
## -- 2021-10-18  1.2.4     DA       Refactoring 
## -- 2021-11-15  1.3.0     DA       Refactoring 
## -- 2021-12-03  1.3.1     DA       Refactoring 
## -- 2021-12-07  1.3.2     DA       Refactoring 
## -- 2022-07-20  1.3.3     SY       Update due to the latest introduction of Gym 0.25
## -- 2022-10-13  1.3.4     SY       Refactoring 
## -- 2022-11-01  1.3.5     DA       Refactoring 
## -- 2022-11-02  1.3.6     DA       Refactoring 
## -- 2022-11-07  1.4.0     DA       Refactoring 
## -- 2023-01-14  1.4.1     MRD      Removing default parameter new_step_api and render_mode for gym
## -------------------------------------------------------------------------------------------------

"""
Ver. 1.4.1 (2023-01-14)

This module shows how to train an agent with a custom policy inside on an OpenAI Gym environment using
MLPro framework.

You will learn:
    
1) How to set up a native policy for an agent
    
2) How to set up an agent
    
3) How to set up a scenario
    
4) How to wrap Gym environment to MLPro environment

5) How to run the scenario and train the agent
    
"""


from mlpro.bf.math import *
from mlpro.rl import *
from mlpro.wrappers.openai_gym import WrEnvGYM2MLPro
import gym
import random
from pathlib import Path




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


    def _adapt(self, p_sars_elem:SARSElement) -> bool:
        # 1.4 Adapting the internal policy is up to you...
        self.log(self.C_LOG_TYPE_I, 'Sorry, I am a stupid agent...')

        # 1.5 Only return True if something has been adapted...
        return False




# 2 Implement your own RL scenario
class MyScenario (RLScenario):

    C_NAME      = 'Matrix'

    def _setup(self, p_mode, p_ada: bool, p_visualize: bool, p_logging) -> Model:
        # 2.1 Setup environment
        gym_env     = gym.make('CartPole-v1')
        self._env   = WrEnvGYM2MLPro(gym_env, p_visualize=p_visualize, p_logging=p_logging) 

        # 2.2 Setup and return standard single-agent with own policy
        return Agent(
                p_policy=MyPolicy(
                    p_observation_space=self._env.get_state_space(),
                    p_action_space=self._env.get_action_space(),
                    p_buffer_size=10,
                    p_ada=p_ada,
                    p_visualize=p_visualize,
                    p_logging=p_logging
                ),    
            p_envmodel=None,
            p_name='Smith',
            p_ada=p_ada,
            p_visualize=p_visualize,
            p_logging=p_logging
        )




# 3 Create scenario and start training

if __name__ == "__main__":
    # 3.1 Parameters for demo mode
    cycle_limit = 500
    logging     = Log.C_LOG_WE
    visualize   = True
    path        = str(Path.home())
 
else:
    # 3.2 Parameters for internal unit test
    cycle_limit = 50
    logging     = Log.C_LOG_NOTHING
    visualize   = False
    path        = None


# 3.3 Create and run training object
training = RLTraining(
        p_scenario_cls=MyScenario,
        p_cycle_limit=cycle_limit,
        p_path=path,
        p_visualize=visualize,
        p_logging=logging )

training.run()