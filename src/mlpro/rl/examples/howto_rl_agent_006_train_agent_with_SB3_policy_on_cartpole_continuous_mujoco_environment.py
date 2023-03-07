## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - A Synoptic Framework for Standardized Machine Learning Tasks
## -- Package : mlpro.rl.examples
## -- Module  : howto_rl_agent_006_train_agent_with_SB3_policy_on_cartpole_continuous_mujoco_environment.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2023-03-07  0.0.0     MRD      Creation
## -- 2023-03-07  1.0.0     MRD      Released first version
## -------------------------------------------------------------------------------------------------

"""
Ver. 1.0.0 (2023-03-07)

This module shows how to train a single agent with SB3 Policy on Cartpole Continuous MuJoCo Environment.
"""


from stable_baselines3 import PPO
from mlpro.rl import *
from mlpro.wrappers.sb3 import WrPolicySB32MLPro
from mlpro.rl.pool.envs.cartpole import CartpoleMujocoContinuous
from pathlib import Path


class MyScenario(RLScenario):
    C_NAME = "Matrix"

    def _setup(self, p_mode, p_ada: bool, p_visualize: bool, p_logging) -> Model:
        # 1.1 Setup environment
        self._env = CartpoleMujocoContinuous(p_logging=logging, p_visualize=visualize)

        # 1.2 Setup Policy From SB3
        policy_sb3 = PPO(policy="MlpPolicy", n_steps=10, env=None, _init_setup_model=False, device="cpu", seed=1)

        # 1.3 Wrap the policy
        policy_wrapped = WrPolicySB32MLPro(
            p_sb3_policy=policy_sb3,
            p_cycle_limit=self._cycle_limit,
            p_observation_space=self._env.get_state_space(),
            p_action_space=self._env.get_action_space(),
            p_ada=p_ada,
            p_visualize=p_visualize,
            p_logging=p_logging,
        )

        # 1.4 Setup standard single-agent with own policy
        return Agent(
            p_policy=policy_wrapped, p_envmodel=None, p_name="Smith", p_ada=p_ada, p_visualize=p_visualize, p_logging=p_logging
        )


# 3 Create scenario and run some cycles
if __name__ == "__main__":
    # Parameters for demo mode
    cycle_limit = 10000
    adaptation_limit = 0
    stagnation_limit = 0
    eval_frequency = 0
    eval_grp_size = 0
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
    p_logging=logging,
)


# 3 Training
training.run()
