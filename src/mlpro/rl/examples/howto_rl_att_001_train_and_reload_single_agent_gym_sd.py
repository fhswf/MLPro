## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - A Synoptic Framework for Standardized Machine Learning Tasks
## -- Package : mlpro.rl.examples
## -- Module  : howto_rl_att_001_train_and_reload_single_agent_gym_sd.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2023-03-04  1.0.0     DA       Creation as derivate of howto_rl_agent_011
## -- 2023-03-27  1.0.1     DA       Refactoring
## -------------------------------------------------------------------------------------------------

"""
Ver. 1.0.1 (2023-03-27)

As in Howto RL AGENT 011, this module shows how to train a single agent and load it again to do some
extra cycles. In opposite to howto 011, stagnation detection is used to automatically end the
training if no further progress can be made.

You will learn:

1. How to use the RLScenario class of MLPro.

2. How to save a scenario after some run.

3. How to reload the saved scenario and re-run for additional cycles.

4. How to use stagnation detection to end the training automatically if there is no progress.

"""


import gym
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
            n_steps=10,
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



if __name__ == '__main__':
    # Parameters for demo mode
    cycle_limit = 20000
    adaptation_limit = 0
    stagnation_limit = 5
    eval_frequency = 10
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
    path = str(Path.home())


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



# 3 Training
training.run()
filename_scenario = training.get_scenario().get_filename()



# 4 Reload the scenario
if __name__ == '__main__':
    input( '\nTraining finished. Press ENTER to reload and run the scenario...\n')

scenario = MyScenario.load( p_path = training.get_training_path() + os.sep + 'scenario',
                            p_filename = filename_scenario )


# 5 Reset Scenario
scenario.reset()  


# 6 Run Scenario
scenario.run()

if __name__ != '__main__':
    from shutil import rmtree
    rmtree(training.get_training_path())
else:
    input( '\nPress ENTER to finish...')
