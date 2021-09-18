## -------------------------------------------------------------------------------------------------
## -- Project : FH-SWF Automation Technology - Common Code Base (CCB)
## -- Package : mlpro.pool.policies
## -- Module  : a2c
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2021-09-18  0.0.0     MRD      Creation
## -- 2021-09-18  1.0.0     MRD      Release first version only for continous action
## -------------------------------------------------------------------------------------------------
## -- Reference
## -- https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail

"""
Ver. 1.0.0 (2021-09-18)

This module provide A2C Algorithm based on reference.
"""

import torch
import random
import torch.optim as optim
from mlpro.rl.models import *
import numpy as np

def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module

class AddBias(torch.nn.Module):
    def __init__(self, bias):
        super(AddBias, self).__init__()
        self._bias = torch.nn.Parameter(bias.unsqueeze(1))

    def forward(self, x):
        bias = self._bias.t().view(1, -1)    
        
        return x + bias

class FixedNormal(torch.distributions.Normal):
    def log_probs(self, actions):
        return super().log_prob(actions).sum(-1, keepdim=True)

    def entrop(self):
        return super.entropy().sum(-1)

    def mode(self):
        return self.mean

class DiagGaussian(torch.nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super(DiagGaussian, self).__init__()

        init_ = lambda m: init(m, torch.nn.init.orthogonal_, lambda x: torch.nn.init.
                               constant_(x, 0))

        self.fc_mean = init_(torch.nn.Linear(num_inputs, num_outputs))
        self.logstd = AddBias(torch.zeros(num_outputs))

    def forward(self, x):
        action_mean = self.fc_mean(x)
        zeros = torch.zeros(action_mean.shape[1])
        action_logstd = self.logstd(zeros)
        
        return action_mean, FixedNormal(action_mean, action_logstd.exp())
    
class MLPBase(torch.nn.Module):
    def __init__(self, num_inputs, hidden_size, recurrent=False):
        super(MLPBase, self).__init__()

        init_ = lambda m: init(m, torch.nn.init.orthogonal_, lambda x: torch.nn.init.
                               constant_(x, 0), np.sqrt(2))

        self.actor = torch.nn.Sequential(
            init_(torch.nn.Linear(num_inputs, hidden_size)), torch.nn.Tanh(),
            init_(torch.nn.Linear(hidden_size, hidden_size)), torch.nn.Tanh())

        self.critic = torch.nn.Sequential(
            init_(torch.nn.Linear(num_inputs, hidden_size)), torch.nn.Tanh(),
            init_(torch.nn.Linear(hidden_size, hidden_size)), torch.nn.Tanh())
        
        self.actor_linear = init_(torch.nn.Linear(hidden_size, 2))
        self.critic_linear = init_(torch.nn.Linear(hidden_size, 1))

        self.train()

    def forward(self, inputs):
        x = inputs
        hidden_critic = self.critic(x)
        hidden_actor = self.actor(x)
        return self.critic_linear(hidden_critic), hidden_actor

class MLPActor(torch.nn.Module):
    def __init__(self, num_inputs, hidden_size, recurrent=False):
        super(MLPActor, self).__init__()

        init_ = lambda m: init(m, torch.nn.init.orthogonal_, lambda x: torch.nn.init.
                               constant_(x, 0), np.sqrt(2))

        self.actor = torch.nn.Sequential(
            init_(torch.nn.Linear(num_inputs, hidden_size)), torch.nn.Tanh(),
            init_(torch.nn.Linear(hidden_size, hidden_size)), torch.nn.Tanh())
        
        self.actor_linear = init_(torch.nn.Linear(hidden_size, 2))

        self.train()

    def forward(self, inputs):
        x = inputs
        hidden_actor = self.actor(x)
        return hidden_actor

class ActorGaussianMLP(torch.nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_size=64):
        super(ActorGaussianMLP, self).__init__()
        self.base = MLPActor(num_inputs, hidden_size)
        self.dist = DiagGaussian(hidden_size,num_actions)
        self.num_inputs = num_inputs
        self.num_actions = num_actions
    
    def sample_action(self, inputs, deterministic=False):
        actor_features = self.base(inputs)
        mean, dist = self.dist(actor_features)

        if deterministic:
            action = dist.mode()
        else:
            action = dist.sample()

        action_log_probs = dist.log_probs(action)

        return action, action_log_probs, mean
    
    def evaluate_action(self, inputs, action):
        actor_features = self.base(inputs)
        mean, dist = self.dist(actor_features)

        action_log_probs = dist.log_probs(action)

        return action_log_probs, mean

class ActorCriticMLP(torch.nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_size=64):
        super(ActorCriticMLP, self).__init__()
        self.base = MLPBase(num_inputs, hidden_size)
        self.dist = DiagGaussian(hidden_size,num_actions)
        self.num_inputs = num_inputs
        self.num_actions = num_actions
    
    def sample_action(self, inputs, deterministic=False):
        value, actor_features = self.base(inputs)
        mean, dist = self.dist(actor_features)

        if deterministic:
            action = dist.mode()
        else:
            action = dist.sample()

        action_log_probs = dist.log_probs(action)
        dist_entropy = dist.entropy().mean()

        return value, action, action_log_probs, dist_entropy
    
    def get_value(self, inputs):
        value, _ = self.base(inputs)
        return value
    
    def evaluate_action(self, inputs, action):
        value, actor_features = self.base(inputs)
        mean, dist = self.dist(actor_features)

        action_log_probs = dist.log_probs(action)
        dist_entropy = dist.entropy().mean()

        return value, action_log_probs, dist_entropy

class A2C(Policy):
    """
    Implementation of A2C Policy Algorithm
    """

    C_NAME = 'A2C'
    
    def __init__(self, p_state_space: MSpace, p_action_space: MSpace, p_buffer_size: int, 
        p_ada, p_use_gae=False, p_gae_lambda=0, p_gamma=0.99, p_value_loss_coef=0.5, p_entropy_coef=0, p_learning_rate=3e-4, p_logging=True):
        """
        Parameters:
            p_state_space (MSpace): State Space
            p_action_space (MSpace): Action Space
            p_buffer_size (int): Buffer size
            p_ada ([type]): Adaptivity
            p_use_gae (bool, optional): Toggle for using GAE. Defaults to False.
            p_gae_lambda (int, optional): GAE Lamda. Defaults to 0.
            p_gamma (float, optional): Gamma. Defaults to 0.99.
            p_value_loss_coef (float, optional): Coefficent for value loss. Defaults to 0.5.
            p_entropy_coef (int, optional): Coefficient for entropy loss. Defaults to 0.
            p_learning_rate ([type], optional): Learning rate. Defaults to 3e-4.
            p_logging (bool, optional): Toggle for logging. Defaults to True.
        """
        super().__init__(p_state_space, p_action_space, p_buffer_size=p_buffer_size, p_ada=p_ada, p_logging=p_logging)
        
        self.use_gae = p_use_gae
        self.gae_lambda = p_gae_lambda
        self.gamma = p_gamma
        self.value_loss_coef = p_value_loss_coef
        self.entropy_coef = p_entropy_coef
        self.learning_rate = p_learning_rate
        self.policy = ActorCriticMLP(self.get_state_space().get_num_dim(),self.get_action_space().get_num_dim())
        self.optimizer = optim.Adam(self.policy.parameters(), lr=self.learning_rate)
        self.additional_buffer_element = {}

    def adapt(self, *p_args) -> bool:
        if not super().adapt(*p_args):
            return False

        # Adapt only when Buffer is full
        if not self._buffer.is_full():
            self.log(self.C_LOG_TYPE_I, 'Buffer is not full yet, keep collecting data!')
            return False
        
        self.log(self.C_LOG_TYPE_I, 'Buffer is full, Adapting Policy!')
    
        # Get All Data From Buffer
        sar_data = self._buffer.get_all()
        
        # Remap the data from the buffer to its own variable
        states = torch.Tensor([state.get_values() for state in sar_data["previous_state"]])
        actions = torch.Tensor([action.get_sorted_values() for action in sar_data["action"]])
        rewards = torch.Tensor([reward.get_overall_reward() for reward in sar_data["reward"]])
        values = torch.Tensor([value for value in sar_data["value"]])
        dones = torch.Tensor([done for done in sar_data["done"]])
        returns = torch.zeros(rewards.size(0))  

        # Get the next value from the last observation
        with torch.no_grad():
            next_value = self.policy.get_value(states[-1]).detach()

        # Calculate Returns
        if self.use_gae:
            returns[-1] = next_value
            gae = 0
            for step in reversed(range(rewards.size(0))):
                if step == self._buffer._size - 1:
                    next_non_terminal = 1.0 - dones[-1]
                    next_values = next_value
                    next_return = returns[-1]
                else:
                    next_non_terminal = 1.0 - dones[step + 1]
                    next_values = values[step + 1]
                    next_return = returns[step + 1]

                delta = rewards[step] + self.gamma * next_values * next_non_terminal - values[step]
                gae = delta + self.gamma * self.gae_lambda * next_non_terminal * gae
                returns[step] = gae + values[step]
        else:
            returns[-1] = next_value
            for step in reversed(range(rewards.size(0))):
                if step == self._buffer._size - 1:
                    next_non_terminal = 1.0 - dones[-1]
                    next_values = next_value
                    next_return = returns[-1]
                else:
                    next_non_terminal = 1.0 - dones[step + 1]
                    next_values = values[step + 1]
                    next_return = returns[step + 1]
                returns[step] = next_return * self.gamma * next_non_terminal + rewards[step]

        # Evaluate the state action pair to get the value
        values, action_log_probs, dist_entropy = self.policy.evaluate_action(states.view(-1,self._state_space.get_num_dim()),actions.view(-1,self._action_space.get_num_dim()))
        
        # Compute Advantage and Value Loss
        values = values.view(self._buffer._size , 1, 1)
        advantages = returns[:-1] - values
        value_loss = advantages.pow(2).mean()

        # Compute Action Loss
        action_log_probs = action_log_probs.view(self._buffer._size , 1, 1)
        action_loss = -(advantages.detach() * action_log_probs).mean()

        # Compute Actor and Critic Loss
        ac_loss = value_loss * self.value_loss_coef + action_loss - dist_entropy * self.entropy_coef
        
        # Update the network
        self.optimizer.zero_grad()
        ac_loss.backward()
        self.optimizer.step()

        # Clear Buffer After Update
        self.clear_buffer()

        return True

    def clear_buffer(self):
        self._buffer.clear()

    def compute_action(self, p_state: State) -> Action:
        obs = p_state.get_values()
        
        if not isinstance(obs, torch.Tensor):
            obs = torch.Tensor(obs).reshape(1,obs.size)
        
        with torch.no_grad():
            value, action, action_log, dist_entropy = self.policy.sample_action(obs,deterministic=True)
        
        # Add to additional_buffer_element
        self.additional_buffer_element = dict(value=value, action_log=action_log, entropy=dist_entropy)

        action = action.cpu().numpy().flatten()
        action = Action(self._id, self._action_space, action)
        return action
    
    def _add_additional_buffer(self, p_buffer_element: SARBufferElement):
        p_buffer_element.add_value_element(self.additional_buffer_element)
        return p_buffer_element
