## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - A Synoptic Framework for Standardized Machine Learning Tasks
## -- Package : mlpro
## -- Module  : howto_rl_002_run_agent_with_own_policy_with_gym_environment.py
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
## -------------------------------------------------------------------------------------------------

"""
Ver. 1.2.2 (2021-12-03)

This module shows how to run an own policy inside the standard agent model with an OpenAI Gym environment using 
the fhswf_at_ml framework.
"""


from mlpro.bf.math import *
from mlpro.rl.models import *
from mlpro.wrappers.openai_gym import WrEnvGYM2MLPro
import gym
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
        # 1 Setup environment
        gym_env     = gym.make('CartPole-v1')
        self._env   = WrEnvGYM2MLPro(gym_env, p_logging=p_logging) 

        # 2 Setup standard single-agent with own policy
        return Agent( p_policy=MyPolicy( p_observation_space=self._env.get_state_space(),
                                         p_action_space=self._env.get_action_space(),
                                         p_buffer_size=1,
                                         p_ada=p_ada,
                                         p_logging=p_logging),    
                      p_envmodel=None,
                      p_name='Smith',
                      p_ada=p_ada,
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