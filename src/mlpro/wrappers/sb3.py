## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - A Synoptic Framework for Standardized Machine Learning Tasks
## -- Package : mlpro.wrappers
## -- Module  : sb3.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2021-10-07  0.0.0     MRD      Creation
## -- 2021-10-07  1.0.0     MRD      Release of first version
## -- 2021-10-18  1.0.1     DA       Refactoring class WrPolicySB32MLPro
## -- 2021-10-18  1.1.0     MRD      SB3 Off Policy Wrapper on WrPolicySB32MLPro
## -- 2021-10-27  1.1.1     MRD      Mismatch datatype last_done on WrPolicySB32MLPro
## -- 2021-11-18  1.1.2     MRD      Put DummyEnv class outside the WrPolicySB32MLPro
## -- 2021-12-20  1.1.3     DA       Replaced calls get_done() by get_success()
## -- 2021-12-21  1.1.4     MRD      Refactor due to new update, regarding initial and terminal
## --                                state 
## -- 2022-01-18  1.1.5     MRD      Fix mismatch Off Policy Algorithm by adding more additional
## --                                information on adapt_off_policy()
## -- 2022-01-20  1.1.6     MRD      Fix the bug due to new version of SB3 1.4.0
## -- 2022-02-25  1.1.7     SY       Refactoring due to auto generated ID in class Dimension
## -- 2022-05-31  1.1.8     SY       Enable the possibility to process reward type C_TYPE_EVERY_AGENT
## -------------------------------------------------------------------------------------------------

"""
Ver. 1.1.8 (2022-05-31)
This module provides wrapper classes for reinforcement learning tasks.
"""

import gym
import torch
from stable_baselines3.common import utils
from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm
from mlpro.rl.models import *


## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class DummyEnv(gym.Env):
    """
    Dummy class for Environment. This is required due to some of the SB3 Policy Algorithm requires to have
    an Environment. As for now, it only needs the observation space and the action space.
    """

    def __init__(self, p_observation_space, p_action_space) -> None:
        super().__init__()
        self.observation_space = p_observation_space
        self.action_space = p_action_space


## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class WrPolicySB32MLPro(Policy):
    """
    This class provides a policy wrapper from Standard Baselines 3 (SB3).
    Especially On-Policy Algorithm
    """

    C_TYPE = 'SB3 Policy'

    ## -------------------------------------------------------------------------------------------------
    def __init__(self, p_sb3_policy, p_cycle_limit, p_observation_space, p_action_space, p_ada=True, p_logging=True):
        """
        Args:
            p_sb3_policy : SB3 Policy
            p_observation_space : Observation Space
            p_action_space : Environment Action Space
            p_ada (bool, optional): Adaptability. Defaults to True.
            p_logging (bool, optional): Logging. Defaults to True.
        """

        super().__init__(p_observation_space, p_action_space, p_ada=p_ada, p_logging=p_logging)

        self.sb3 = p_sb3_policy
        self.last_buffer_element = None
        self.last_done = False

        # Variable preparation for SB3
        action_space = None
        observation_space = None

        # Check if action is Discrete or Box
        action_dim = self.get_action_space().get_num_dim()
        id_dim = self.get_action_space().get_dim_ids()[0]
        if len(self.get_action_space().get_dim(id_dim).get_boundaries()) == 1:
            action_space = gym.spaces.Discrete(self.get_action_space().get_dim(id_dim).get_boundaries()[0])
        else:
            self.lows = []
            self.highs = []
            for dimension in range(action_dim):
                id_dim = self.get_action_space().get_dim_ids()[dimension]
                self.lows.append(self.get_action_space().get_dim(id_dim).get_boundaries()[0])
                self.highs.append(self.get_action_space().get_dim(id_dim).get_boundaries()[1])

            action_space = gym.spaces.Box(
                low=np.array(self.lows, dtype=np.float32),
                high=np.array(self.highs, dtype=np.float32),
                shape=(action_dim,),
                dtype=np.float32
            )

        # Check if state is Discrete or Box
        observation_dim = self.get_observation_space().get_num_dim()
        id_dim = self.get_observation_space().get_dim_ids()[0]
        if len(self.get_observation_space().get_dim(id_dim).get_boundaries()) == 1:
            observation_space = gym.spaces.Discrete(self.get_observation_space().get_dim(id_dim).get_boundaries()[0])
        else:
            lows = []
            highs = []
            for dimension in range(observation_dim):
                id_dim = self.get_observation_space().get_dim_ids()[dimension]
                lows.append(self.get_observation_space().get_dim(id_dim).get_boundaries()[0])
                highs.append(self.get_observation_space().get_dim(id_dim).get_boundaries()[1])

            observation_space = gym.spaces.Box(
                low=np.array(lows, dtype=np.float32),
                high=np.array(highs, dtype=np.float32),
                shape=(observation_dim,),
                dtype=np.float32
            )

        # Create Dummy Env
        self.sb3.env = DummyEnv(observation_space, action_space)

        # Setup SB3 Model
        self.sb3.observation_space = observation_space
        self.sb3.action_space = action_space
        self.sb3.n_envs = 1

        if isinstance(p_sb3_policy, OnPolicyAlgorithm):
            self.compute_action = self._compute_action_on_policy
            self._add_buffer = self._add_buffer_on_policy
            self._adapt = self._adapt_on_policy
            self.clear_buffer = self._clear_buffer_on_policy
            self._buffer = self.sb3.rollout_buffer
        else:
            self.compute_action = self._compute_action_off_policy
            self._add_buffer = self._add_buffer_off_policy
            self._adapt = self._adapt_off_policy
            self.clear_buffer = self._clear_buffer_off_policy
            self._buffer = self.sb3.replay_buffer
            self.collected_steps = 0

        self.sb3._setup_model()
        self.sb3._total_timesteps = p_cycle_limit
        self.sb3._logger = utils.configure_logger()

    def _compute_action_on_policy(self, p_obs: State) -> Action:
        obs = p_obs.get_values()

        if self._adaptivity:
            if not isinstance(obs, torch.Tensor):
                if isinstance(obs, list):
                    obs = torch.Tensor(obs).reshape(1, len(obs)).to(self.sb3.device)
                else:
                    obs = torch.Tensor(obs).reshape(1, obs.size).to(self.sb3.device)

            with torch.no_grad():
                actions, values, log_probs = self.sb3.policy.forward(obs)

            actions = actions.cpu().numpy()

            clipped_actions = actions
            # Clip the actions to avoid out of bound error
            if isinstance(self.sb3.action_space, gym.spaces.Box):
                clipped_actions = np.clip(actions, self.lows, self.highs)

            # Action Step
            action = clipped_actions.flatten()
            action = Action(self._id, self._action_space, action)

            # Action Buffer
            action_buffer = actions.flatten()
            action_buffer = Action(self._id, self._action_space, action_buffer)

            # Add to additional_buffer_element
            self.additional_buffer_element = dict(action=action_buffer, value=values, action_log=log_probs)
        else:
            if isinstance(obs, list):
                obs = torch.Tensor(obs).reshape(1, len(obs)).to(self.sb3.device)
            else:
                obs = torch.Tensor(obs).reshape(1, obs.size).to(self.sb3.device)
            action, _ = self.sb3.predict(obs, deterministic=True)

            action = action.flatten()
            action = Action(self._id, self._action_space, action)
        return action

    def _compute_action_off_policy(self, p_obs: State) -> Action:
        self.sb3._last_obs = p_obs.get_values()
        action, buffer_action = self.sb3._sample_action(self.sb3.learning_starts)

        action = action.flatten()
        action = Action(self._id, self._action_space, action)

        buffer_action = buffer_action.flatten()
        buffer_action = Action(self._id, self._action_space, buffer_action)

        # Add to additional_buffer_element
        self.additional_buffer_element = dict(action=buffer_action)
        return action

    def _adapt_off_policy(self, *p_args) -> bool:
        # Add to buffer
        self._add_buffer(p_args[0])

        # Should Collect more steps
        if self.collected_steps < self.sb3.train_freq.frequency:
            return False

        # Check Step > Warp Up Phase
        if self.sb3.num_timesteps > 0 and self.sb3.num_timesteps > self.sb3.learning_starts:
            gradient_steps = self.sb3.gradient_steps if self.sb3.gradient_steps >= 0 else self.collected_steps
            if gradient_steps > 0:
                self.sb3.train(batch_size=self.sb3.batch_size, gradient_steps=gradient_steps)
                self.collected_steps = 0
                return True

        self.collected_steps = 0
        return False

    def _adapt_on_policy(self, *p_args) -> bool:
        # Add to buffer
        self._add_buffer(p_args[0])

        # Adapt only when Buffer is full
        if not self.sb3.rollout_buffer.full:
            self.log(self.C_LOG_TYPE_I, 'Buffer is not full yet, keep collecting data!')
            return False

        last_obs = torch.Tensor(np.array([self.last_buffer_element.get_data()["state_new"].get_values()])).to(
            self.sb3.device)
        last_done = self.last_buffer_element.get_data()["state_new"].get_terminal()
        last_done = np.array([last_done])

        # Get the next value from the last observation
        with torch.no_grad():
            last_values = self.sb3.policy.predict_values(last_obs)

        # Compute Return and Advantage
        self.sb3.rollout_buffer.compute_returns_and_advantage(last_values=last_values, dones=last_done)

        # Train
        self.sb3.train()

        # Clear Buffer After Update
        self.sb3.rollout_buffer.reset()

        return True

    def _clear_buffer_on_policy(self):
        self.sb3.rollout_buffer.reset()

    def _clear_buffer_off_policy(self):
        self.sb3.replay_buffer.reset()

    def _add_buffer_off_policy(self, p_buffer_element: SARSElement):
        """
        Redefine add_buffer function. Instead of adding to MLPro SARBuffer, we are using
        internal buffer from SB3 for off_policy.
        """

        # Add num_collected_steps
        self.collected_steps += 1
        self.sb3.num_timesteps += 1

        self.last_buffer_element = self._add_additional_buffer(p_buffer_element)
        datas = self.last_buffer_element.get_data()

        info = {}

        if datas["state_new"].get_terminal() and datas["state_new"].get_timeout():
            info["TimeLimit.truncated"] = True

        self.sb3.replay_buffer.add(
            datas["state"].get_values(),
            datas["state_new"].get_values(),
            datas["action"].get_sorted_values(),
            datas["reward"].get_overall_reward(),
            datas["state_new"].get_terminal(),
            [info])

        self.sb3._update_current_progress_remaining(self.sb3.num_timesteps, self.sb3._total_timesteps)
        self.sb3._on_step()

    def _add_buffer_on_policy(self, p_buffer_element: SARSElement):
        """
        Redefine add_buffer function. Instead of adding to MLPro SARBuffer, we are using
        internal buffer from SB3 for on_policy.
        """
        self.sb3.num_timesteps += 1
        self.last_buffer_element = self._add_additional_buffer(p_buffer_element)
        datas = self.last_buffer_element.get_data()

        try:        
            rewards = datas["reward"].get_overall_reward()
        except:
            rewards = datas["reward"].get_agent_reward(self._id)

        if datas["state_new"].get_terminal() and datas["state_new"].get_timeout():
            terminal_obs = torch.Tensor(np.array([datas["state_new"].get_values()])).to(self.sb3.device)
            with torch.no_grad():
                terminal_value = self.sb3.policy.predict_values(terminal_obs)[0]
            rewards += self.sb3.gamma * terminal_value.cpu()

        self.sb3.rollout_buffer.add(
            datas["state"].get_values(),
            datas["action"].get_sorted_values(),
            rewards,
            datas["state"].get_initial(),
            datas["value"],
            datas["action_log"])

    ## -------------------------------------------------------------------------------------------------
    def _add_additional_buffer(self, p_buffer_element: SARSElement):
        p_buffer_element.add_value_element(self.additional_buffer_element)
        return p_buffer_element
