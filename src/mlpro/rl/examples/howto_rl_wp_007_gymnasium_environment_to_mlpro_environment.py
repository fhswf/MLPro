## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - A Synoptic Framework for Standardized Machine Learning Tasks
## -- Package : mlpro.rl.examples
## -- Module  : howto_rl_wp_007_gymnasium_environment_to_mlpro_environment.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2023-04-10  0.0.0     MRD      Creation
## -- 2023-04-10  1.0.0     MRD      First Release
## -------------------------------------------------------------------------------------------------

"""
Ver. 1.0.0 (2023-04-10)

This module shows how to wrap a native MLPro environment class to Gym environment based on Gymnasium.

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
        self._env   = WrEnvGYM2MLPro( p_gym_env_id='CartPole-v1', p_visualize=p_visualize, p_logging=p_logging) 

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
