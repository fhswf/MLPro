## -------------------------------------------------------------------------------------------------
## -- Project : FH-SWF Automation Technology - Common Code Base (CCB)
## -- Package : mlpro
## -- Module  : test_policy_wrapper
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2021-09-11  1.0.0     MRD      Creation
## -- 2021-09-21  1.0.0     MRD      Release First Version
## -- 2021-10-27  1.0.1     MRD      Added Policy Loss Check between Native and Wrapper
## -- 2021-12-08  1.0.2     DA       Refactoring
## -- 2021-12-20  1.0.3     DA       Refactoring
## -- 2022-01-18  2.0.0     MRD      Add Off Policy Algorithm into the test
## -- 2022-01-21  2.0.1     MRD      Include RobotHTM as the continues action envrionment
## -------------------------------------------------------------------------------------------------

"""
Ver. 2.0.1 (2022-01-21)

Unit test classes for environment.
"""

from numpy import empty
import pytest
import gym
import torch
from mlpro.rl.models import *
from mlpro.wrappers.openai_gym import WrEnvGYM2MLPro, WrEnvMLPro2GYM
from mlpro.rl.pool.envs.robotinhtm import RobotHTM
from mlpro.wrappers.sb3 import WrPolicySB32MLPro
from stable_baselines3 import A2C, PPO, DQN, DDPG, SAC
from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm
from stable_baselines3.common.callbacks import BaseCallback


## -------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("env_cls", [PPO, A2C, DQN, DDPG, SAC])
def test_sb3_policy_wrapper(env_cls):
    buffer_size = 100
    policy_kwargs_on = dict(activation_fn=torch.nn.Tanh,
                            net_arch=[dict(pi=[128, 128], vf=[128, 128])])

    policy_kwargs_off = dict(activation_fn=torch.nn.ReLU,
                             net_arch=[10])

    class MyScenario(RLScenario):

        C_NAME = 'Matrix'

        def _setup(self, p_mode, p_ada, p_logging):
            class CustomWrapperFixedSeed(WrEnvGYM2MLPro):
                def _reset(self, p_seed=None):
                    self.log(self.C_LOG_TYPE_I, 'Reset')
                    self._num_cycles = 0

                    # 1 Reset Gym environment and determine initial state
                    observation = self._gym_env.reset()
                    obs = DataObject(observation)

                    # 2 Create state object from Gym observation
                    state = State(self._state_space)
                    state.set_values(obs.get_data())
                    self._set_state(state)

            if issubclass(env_cls, OnPolicyAlgorithm):
                # 1 Setup environment
                self._env = RobotHTM(p_reset_seed=False, p_target_mode="fix", p_logging=False)
                policy_sb3 = env_cls(
                    policy="MlpPolicy",
                    env=None,
                    n_steps=buffer_size,
                    _init_setup_model=False,
                    policy_kwargs=policy_kwargs_on,
                    verbose=0,
                    seed=2)
            else:
                if issubclass(env_cls, DQN):
                    # 1 Setup environment
                    gym_env = gym.make('CartPole-v1')
                    gym_env.seed(2)
                    self._env = CustomWrapperFixedSeed(gym_env, p_logging=False)
                else:
                    self._env = RobotHTM(p_reset_seed=False, p_target_mode="fix", p_logging=False)

                policy_sb3 = env_cls(
                    policy="MlpPolicy",
                    env=None,
                    buffer_size=1000000,
                    _init_setup_model=False,
                    policy_kwargs=policy_kwargs_off,
                    gradient_steps=1,
                    train_freq=4,
                    learning_starts=0,
                    verbose=0,
                    seed=2)

            class TestWrPolicySB32MLPro(WrPolicySB32MLPro):
                """
                Custom Class for logging the loss
                """

                def __init__(self, p_sb3_policy, p_cycle_limit, p_observation_space, p_action_space, p_ada=True,
                             p_logging=True):
                    super().__init__(p_sb3_policy, p_cycle_limit, p_observation_space, p_action_space, p_ada=p_ada,
                                     p_logging=p_logging)
                    self.loss_cnt = []

                def _adapt_off_policy(self, *p_args) -> bool:
                    if super()._adapt_off_policy(*p_args):
                        if isinstance(self.sb3, DQN):
                            self.loss_cnt.append(self.sb3.logger.name_to_value["train/loss"])
                        elif isinstance(self.sb3, DDPG):
                            self.loss_cnt.append(self.sb3.logger.name_to_value["train/critic_loss"])
                        elif isinstance(self.sb3, SAC):
                            self.loss_cnt.append(self.sb3.logger.name_to_value["train/critic_loss"])
                        return True
                    return False

                def _adapt_on_policy(self, *p_args) -> bool:
                    if super()._adapt_on_policy(*p_args):
                        # Log the Loss
                        if isinstance(self.sb3, PPO):
                            self.loss_cnt.append(self.sb3.logger.name_to_value["train/policy_gradient_loss"])
                        elif isinstance(self.sb3, A2C):
                            self.loss_cnt.append(self.sb3.logger.name_to_value["train/policy_loss"])
                        return True
                    return False

            # 3 Wrap the policy
            self.policy_wrapped = TestWrPolicySB32MLPro(
                p_sb3_policy=policy_sb3,
                p_cycle_limit=self._cycle_limit,
                p_observation_space=self._env.get_state_space(),
                p_action_space=self._env.get_action_space(),
                p_ada=p_ada,
                p_logging=p_logging)

            # 4 Setup standard single-agent with own policy
            return Agent(
                p_policy=self.policy_wrapped,
                p_envmodel=None,
                p_name='Smith',
                p_ada=p_ada,
                p_logging=p_logging
            )

    # 3 Instantiate training
    training = RLTraining(
        p_scenario_cls=MyScenario,
        p_cycle_limit=1200,
        p_stagnation_limit=0,
        p_collect_states=True,
        p_collect_actions=True,
        p_collect_rewards=True,
        p_collect_training=True,
        p_logging=False
    )

    # 4 Train
    training.run()

    class CustomCallback(BaseCallback):

        def __init__(self, p_verbose=0):
            super(CustomCallback, self).__init__(p_verbose)
            self.update = 0
            self.loss_cnt = []

        def _on_rollout_start(self) -> None:
            if self.update != 0:
                if isinstance(self.locals.get("self"), PPO):
                    self.loss_cnt.append(self.locals.get("self").logger.name_to_value["train/policy_gradient_loss"])
                elif isinstance(self.locals.get("self"), A2C):
                    self.loss_cnt.append(self.locals.get("self").logger.name_to_value["train/policy_loss"])
                elif isinstance(self.locals.get("self"), DQN):
                    self.loss_cnt.append(self.locals.get("self").logger.name_to_value["train/loss"])
                elif isinstance(self.locals.get("self"), DDPG):
                    self.loss_cnt.append(self.locals.get("self").logger.name_to_value["train/critic_loss"])
                elif isinstance(self.locals.get("self"), SAC):
                    self.loss_cnt.append(self.locals.get("self").logger.name_to_value["train/critic_loss"])

        def _on_step(self) -> bool:
            return super()._on_step()

        def _on_rollout_end(self) -> None:
            self.update += 1

    if issubclass(env_cls, OnPolicyAlgorithm):
        # 1 Setup environment
        env = RobotHTM(p_reset_seed=False, p_target_mode="fix", p_logging=False)
        gym_env = WrEnvMLPro2GYM(env)
        policy_sb3 = env_cls(
            policy="MlpPolicy",
            env=gym_env,
            n_steps=buffer_size,
            verbose=0,
            policy_kwargs=policy_kwargs_on,
            seed=2)
    else:
        if issubclass(env_cls, DQN):
            # 1 Setup environment
            gym_env = gym.make('CartPole-v1')
            gym_env.seed(2)
        else:
            env = RobotHTM(p_reset_seed=False, p_target_mode="fix", p_logging=False)
            gym_env = WrEnvMLPro2GYM(env)
        policy_sb3 = env_cls(
            policy="MlpPolicy",
            env=gym_env,
            buffer_size=1000000,
            verbose=0,
            gradient_steps=1,
            train_freq=4,
            learning_starts=0,
            policy_kwargs=policy_kwargs_off,
            seed=2)

    cus_callback = CustomCallback()
    policy_sb3.learn(total_timesteps=1200, callback=cus_callback)

    assert cus_callback.loss_cnt is not empty, "No Loss on Native"
    assert training.get_scenario().policy_wrapped.loss_cnt is not empty, "No Loss on Wrapper"
    length = min(len(cus_callback.loss_cnt), len(training.get_scenario().policy_wrapped.loss_cnt))
    assert np.linalg.norm(np.array(cus_callback.loss_cnt[:length]) - np.array(
        training.get_scenario().policy_wrapped.loss_cnt[:length])) == 0.0, "Mismatch Native and Wrapper"