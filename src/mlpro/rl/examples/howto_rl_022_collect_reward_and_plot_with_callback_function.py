## -------------------------------------------------------------------------------------------------
## -- Project : FH-SWF Automation Technology - Common Code Base (CCB)
## -- Package : mlpro
## -- Module  : howto_rl_022_collect_reward_and_plot_with_callback_function.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2022-09-02  0.0.0     MRD       Creation
## -------------------------------------------------------------------------------------------------


"""
Ver. 0.0.0 (2022-09-02)

This module shows how to collect reward and plot the reward from custom callback.
"""


from mlpro.bf.math import *
from mlpro.rl.models import *
from mlpro.wrappers.openai_gym import WrEnvGYM2MLPro
import gym
import random
from pathlib import Path
import matplotlib.pyplot as plt



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
        self.log(self.C_LOG_TYPE_I, 'Sorry, I am a stupid agent...')

        # 2 Only return True if something has been adapted...
        return False

    

# 2 Implement Custom Callback
class MyCallback(RLCallback):
    
    C_NAME      = "MyCallback"

    def _init_callback(self):
        self.reward_container = []

    def _episode_start(self):
        self.total_reward_episode = 0

    def _after_process_action(self):
        self.total_reward_episode += self.locals["reward"].get_overall_reward()

    def _episode_end(self):
        self.reward_container.append(self.total_reward_episode)
        self.total_reward_episode = 0

    def _training_end(self):
        plt.plot(self.reward_container)
        plt.ylabel('Reward')
        plt.show()



# 3 Implement your own RL scenario
class MyScenario (RLScenario):

    C_NAME      = 'Matrix'

    def _setup(self, p_mode, p_ada, p_logging):
        # 1 Setup environment
        gym_env     = gym.make('CartPole-v1', new_step_api=True, render_mode=None)
        self._env   = WrEnvGYM2MLPro(gym_env, p_logging=p_logging) 

        # 2 Setup and return standard single-agent with own policy
        return Agent(
                p_policy=MyPolicy(
                    p_observation_space=self._env.get_state_space(),
                    p_action_space=self._env.get_action_space(),
                    p_buffer_size=10,
                    p_ada=p_ada,
                    p_logging=p_logging
                ),    
            p_envmodel=None,
            p_name='Smith',
            p_ada=p_ada,
            p_logging=p_logging
        )




# 4 Create scenario and start training

if __name__ == "__main__":
    # 4.1 Parameters for demo mode
    cycle_limit = 500
    logging     = Log.C_LOG_WE
    visualize   = True
    path        = str(Path.home())
 
else:
    # 4.2 Parameters for internal unit test
    cycle_limit = 50
    logging     = Log.C_LOG_NOTHING
    visualize   = False
    path        = None


# 4.3 Create and run training object
training = RLTraining(
        p_scenario_cls=MyScenario,
        p_callback_cls=MyCallback,
        p_cycle_limit=cycle_limit,
        p_path=path,
        p_visualize=visualize,
        p_logging=logging )

training.run()