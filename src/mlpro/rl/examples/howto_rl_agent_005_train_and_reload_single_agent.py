## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - A Synoptic Framework for Standardized Machine Learning Tasks
## -- Package : mlpro.rl.examples
## -- Module  : howto_rl_agent_005_train_and_reload_single_agent.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2022-01-28  0.0.0     MRD      Creation
## -- 2022-01-28  1.0.0     MRD      Released first version
## -- 2022-05-19  1.0.1     MRD      Re-use the agent not for the re-training process
## --                                Remove commenting and numbering
## -- 2022-05-19  1.0.2     MRD      Re-add the commenting and reformat the numbering in comment
## -- 2022-07-20  1.0.3     SY       Update due to the latest introduction of Gym 0.25
## -- 2022-10-13  1.0.4     SY       Refactoring 
## -- 2022-10-17  1.0.5     SY       Debugging 
## -- 2022-11-01  1.0.6     DA       Refactoring
## -- 2022-11-07  1.1.0     DA       Refactoring
## -- 2023-01-14  1.1.1     MRD      Removing default parameter new_step_api and render_mode for gym
## -- 2023-02-12  1.1.2     MRD      Add randomness path for CI test
## -------------------------------------------------------------------------------------------------

"""
Ver. 1.1.2 (2023-02-12)

This module shows how to train a single agent and load it again to do some extra cycles.
"""


import gym
import string
from stable_baselines3 import PPO
from mlpro.rl import *
from mlpro.wrappers.openai_gym import WrEnvGYM2MLPro
from mlpro.wrappers.sb3 import WrPolicySB32MLPro
from pathlib import Path



# 1 Implement your own RL scenario
class MyScenario (RLScenario):
    C_NAME = 'Matrix'

    def _setup(self, p_mode, p_ada: bool, p_visualize: bool, p_logging) -> Model:
        # 1.1 Setup environment
        gym_env = gym.make('CartPole-v1')
        self._env = WrEnvGYM2MLPro(gym_env, p_visualize=p_visualize, p_logging=p_logging)

        # 1.2 Setup Policy From SB3
        policy_sb3 = PPO(
            policy="MlpPolicy",
            n_steps=5,
            env=None,
            _init_setup_model=False,
            device="cpu",
            seed=1)

        # 1.3 Wrap the policy
        policy_wrapped = WrPolicySB32MLPro(
            p_sb3_policy=policy_sb3,
            p_cycle_limit=self._cycle_limit,
            p_observation_space=self._env.get_state_space(),
            p_action_space=self._env.get_action_space(),
            p_ada=p_ada,
            p_visualize=p_visualize,
            p_logging=p_logging)

        # 1.4 Setup standard single-agent with own policy
        return Agent(
            p_policy=policy_wrapped,
            p_envmodel=None,
            p_name='Smith',
            p_ada=p_ada,
            p_visualize=p_visualize,
            p_logging=p_logging
        )



if __name__ == "__main__":
    # Parameters for demo mode
    cycle_limit = 5000
    adaptation_limit = 50
    stagnation_limit = 5
    eval_frequency = 5
    eval_grp_size = 5
    logging = Log.C_LOG_WE
    visualize = True
    path = str(Path.home())

else:
    # Parameters for internal unit test
    cycle_limit = 50
    adaptation_limit = 5
    stagnation_limit = 5
    eval_frequency = 2
    eval_grp_size = 1
    logging = Log.C_LOG_NOTHING
    visualize = False
    path = str(Path.home()) + str(os.sep) + ''.join(random.choice(string.ascii_lowercase) for i in range(5))


# 2 Create scenario and start training
training = RLTraining(
    p_scenario_cls=MyScenario,
    p_cycle_limit=cycle_limit,
    p_adaptation_limit=adaptation_limit,
    p_stagnation_limit=stagnation_limit,
    p_eval_frequency=eval_frequency,
    p_eval_grp_size=eval_grp_size,
    p_path=path,
    p_visualize=visualize,
    p_logging=logging )


# 3 Create scenario and start training
training.run()


# 4 Save the training path for loading the agent model file
training_path = training._root_path


# 
# Now we start from the beginning. This time we load an existing model.
#

# 5 Implement your own RL scenario with an existing model
class MyNdScenario(RLScenario):
    C_NAME = 'Matrix2'

    def _setup(self, p_mode, p_ada: bool, p_visualize: bool, p_logging) -> Model:
        # 5.1 Setup environment
        gym_env = gym.make('CartPole-v1')
        self._env = WrEnvGYM2MLPro(gym_env, p_visualize=p_visualize, p_logging=p_logging)

        # 5.2 In this example we use previous training from the same file
        # To make easier, we retrieve the save path from the previous training
        return self.load(training_path, "trained model.pkl")


# 6 Instatiate new scenario
scenario = MyNdScenario(p_mode=Mode.C_MODE_SIM, 
                        p_ada=False,
                        p_cycle_limit=cycle_limit,
                        p_visualize=visualize,
                        p_logging=logging)


# 7 Reset Scenario
scenario.reset()  


# 8 Run Scenario
scenario.run()
