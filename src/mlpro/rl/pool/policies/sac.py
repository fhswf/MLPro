## -------------------------------------------------------------------------------------------------
## -- Project : FH-SWF Automation Technology - Common Code Base (CCB)
## -- Package : mlpro.pool.policies
## -- Module  : sac
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2021-09-24  0.0.0     MRD      Creation
## -- 2021-09-25  1.0.0     MRD      Release first version
## -- 2021-09-26  1.0.0     MRD      Change the exploration to warm up phase only
## --                                adjustment on the critic loss calculation
## -------------------------------------------------------------------------------------------------
## -- Reference
## -- https://github.com/DLR-RM/stable-baselines3

"""
Ver. 1.0.0 (2021-09-26)

This module provide SAC Algorithm based on reference.
"""

import torch
import random
import torch.optim as optim
from mlpro.rl.models import *
from mlpro.rl.pool.sarbuffer.randomsarbuffer import RandomSARBuffer
import numpy as np

def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module

def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)


## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class AddBias(torch.nn.Module):
    def __init__(self, bias):
        super(AddBias, self).__init__()
        self._bias = torch.nn.Parameter(bias.unsqueeze(1))

    def forward(self, x):
        bias = self._bias.t().view(1, -1)    
        
        return x + bias

## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class FixedNormal(torch.distributions.Normal):
    def log_probs(self, actions):
        return super().log_prob(actions).sum(-1, keepdim=True)

    def entropy(self):
        return super().entropy().sum(-1)

    def mode(self):
        return self.mean

## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class FixedCategorical(torch.distributions.Categorical):
    def log_probs(self, actions):
        return super().log_prob(actions).sum(-1, keepdim=True)

    def entropy(self):
        return super().entropy().sum(-1)

    def mode(self):
        return torch.argmax(self.probs, dim=1)

## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class DiagGaussianDistribution(torch.nn.Module):
    """
    Diagonal Gaussian for Continuous Action

    """
    def __init__(self, num_inputs, num_outputs):
        super(DiagGaussianDistribution, self).__init__()

        init_ = lambda m: init(m, torch.nn.init.orthogonal_, lambda x: torch.nn.init.
                               constant_(x, 0))

        self.fc_mean = init_(torch.nn.Linear(num_inputs, num_outputs))
        self.logstd = AddBias(torch.zeros(num_outputs))

    def forward(self, x):
        action_mean = self.fc_mean(x)
        zeros = torch.zeros(action_mean.shape[1])
        action_logstd = self.logstd(zeros)
        
        return action_mean, FixedNormal(action_mean, action_logstd.exp())

## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class CategoricalDistribution(torch.nn.Module):
    """
    Categorical Distribution for Discrete Action

    """
    def __init__(self, num_inputs, num_outputs):
        super(CategoricalDistribution, self).__init__()

        init_ = lambda m: init(m, torch.nn.init.orthogonal_, lambda x: torch.nn.init.
                               constant_(x, 0))

        self.fc_mean = init_(torch.nn.Linear(num_inputs, num_outputs))

    def forward(self, x):
        action_mean = self.fc_mean(x)
        
        return action_mean, FixedCategorical(logits=action_mean)

## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
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

## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class ActorGaussianMLP(torch.nn.Module):
    """
    Class provides Actor MLP

    """
    def __init__(self, num_inputs, num_actions, p_dist_cls, hidden_size=64):
        super(ActorGaussianMLP, self).__init__()
        self.base = MLPActor(num_inputs, hidden_size)
        self.dist = p_dist_cls(hidden_size,num_actions)
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

## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class QNetwork(torch.nn.Module):
    """
    Class provides Single Q Network
    
    """
    def __init__(self, num_inputs, num_actions, hidden_dim):
        super(QNetwork, self).__init__()

        init_ = lambda m: init(m, torch.nn.init.orthogonal_, lambda x: torch.nn.init.
                               constant_(x, 0), np.sqrt(2))

        self.qnet = torch.nn.Sequential(
            init_(torch.nn.Linear(num_inputs + num_actions, hidden_dim)), torch.nn.ReLU(),
            init_(torch.nn.Linear(hidden_dim, hidden_dim)), torch.nn.ReLU())

        self.qnet_linear = init_(torch.nn.Linear(hidden_dim, 1))

    def forward(self, state, action):
        state = state.reshape(state.shape[0],state.shape[-1])
        action = action.reshape(action.shape[0],action.shape[-1])
        
        xu = torch.cat([state, action], 1)
        
        x1 = self.qnet(xu)
        x1 = self.qnet_linear(x1)

        return x1

## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class DQNetwork(torch.nn.Module):
    """
    Class provides Double Q Network

    """
    def __init__(self, num_inputs, num_actions, hidden_dim):
        super(DQNetwork, self).__init__()

        init_ = lambda m: init(m, torch.nn.init.orthogonal_, lambda x: torch.nn.init.
                               constant_(x, 0), np.sqrt(2))

        self.qnet1 = torch.nn.Sequential(
            init_(torch.nn.Linear(num_inputs + num_actions, hidden_dim)), torch.nn.ReLU(),
            init_(torch.nn.Linear(hidden_dim, hidden_dim)), torch.nn.ReLU())

        self.qnet1_linear = init_(torch.nn.Linear(hidden_dim, 1))

        self.qnet2 = torch.nn.Sequential(
            init_(torch.nn.Linear(num_inputs + num_actions, hidden_dim)), torch.nn.ReLU(),
            init_(torch.nn.Linear(hidden_dim, hidden_dim)), torch.nn.ReLU())

        self.qnet2_linear = init_(torch.nn.Linear(hidden_dim, 1))

    def forward(self, state, action):
        state = state.reshape(state.shape[0],state.shape[-1])
        action = action.reshape(action.shape[0],action.shape[-1])
        
        xu = torch.cat([state, action], 1)
        
        x1 = self.qnet1(xu)
        x1 = self.qnet1_linear(x1)

        x2 = self.qnet2(xu)
        x2 = self.qnet2_linear(x2)

        return x1, x2

## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class SAC(Policy):
    """
    Implementation of SAC Policy Algorithm
    """

    C_NAME = 'SAC'
    
    def __init__(self, p_state_space: MSpace, p_action_space: MSpace, p_buffer_size: int, 
                p_ada, p_batch_size=64, p_buffer_cls=RandomSARBuffer, p_explore_chance=0.5, p_qnet_type="Q", 
                p_alpha=0.2, p_tau=0.005, p_gamma=0.99, p_gradient_step=1, p_target_update_interval=1, 
                p_warm_up_step=50000, p_automatic_entropy_tuning=True, p_learning_rate=3e-4, 
                p_actor_lr=0.0003, p_critic_lr=0.0003, p_logging=True):
        """
        Args:
            p_state_space (MSpace): State Space
            p_action_space (MSpace): Action Space
            p_buffer_size (int): Buffer size
            p_ada ([type]): Adaptability
            p_batch_size (int, optional): Batch size per gradient update. Defaults to 64.
            p_buffer_cls ([type], optional): Class type of Buffer. Defaults to RandomSARBuffer.
            p_qnet_type (str, optional): Type of Q network. Q for single Q. DQ for double Q. Defaults to "Q".
            p_alpha (float, optional): Entropy Coefficient. Defaults to 0.2.
            p_tau (float, optional): Soft update coeefficient. Defaults to 0.005.
            p_gamma (float, optional): Discount factor. Defaults to 0.99.
            p_gradient_step (int, optional): Update per rollout. Defaults to 1.
            p_target_update_interval (int, optional): Update interval. Defaults to 1.
            p_automatic_entropy_tuning (bool, optional): Automatic entropy learning. Defaults to True.
            p_learning_rate ([type], optional): Learning. Defaults to 3e-4.
            p_logging (bool, optional): Logging. Defaults to True.
        """
        super().__init__(p_state_space, p_action_space, p_buffer_size=p_buffer_size, 
                        p_buffer_cls=p_buffer_cls, p_ada=p_ada, p_logging=p_logging)
        

        self.batch_size = p_batch_size
        self.warm_up_phase = p_warm_up_step
        self.gradient_step = p_gradient_step
        self.qnet_type = p_qnet_type
        self.automatic_entropy_tuning = p_automatic_entropy_tuning
        self.target_update_interval = p_target_update_interval
        self.tau = p_tau
        self.alpha = p_alpha
        self.gamma = p_gamma
        self.exploration_chance = p_explore_chance
        self.learning_rate = p_learning_rate
        self.actor_learning_rate = p_actor_lr
        self.critic_learning_rate = p_critic_lr
        self.additional_buffer_element = {}
        self._setup_policy()
        
        self.episode_step = 0

## -------------------------------------------------------------------------------------------------
    def _setup_policy(self):
        """
        Setup the Policy Network based on type of the action space of an environment.
        """
        
        action_dim = self.get_action_space().get_num_dim()
        state_dim = self.get_state_space().get_num_dim()
        self._dist_cls = DiagGaussianDistribution

        # Check if action is Discrete
        if self.get_action_space().get_num_dim() == 1:
            if len(self.get_action_space().get_dim(0).get_boundaries()) == 1:
                action_dim = self.get_action_space().get_dim(0).get_boundaries()[0]
                self._dist_cls = CategoricalDistribution


        self.policy = ActorGaussianMLP(state_dim,
                                    action_dim, self._dist_cls)

        self.policy_optim = optim.Adam(self.policy.parameters(), lr=self.actor_learning_rate)

        self.critic = QNetwork(self.get_state_space().get_num_dim(), self.get_action_space().get_num_dim(), 
                                64)

        self.critic_target = QNetwork(self.get_state_space().get_num_dim(), self.get_action_space().get_num_dim(), 
                                64)

        hard_update(self.critic_target, self.critic)

        self.critic_optim = optim.Adam(self.critic.parameters(), lr=self.critic_learning_rate)

        if self.automatic_entropy_tuning is True:
            init_value = 1.0

            # Target Entropy Coef
            self.target_entropy = -torch.prod(torch.Tensor(self.get_action_space().get_num_dim())).item()

            # Log Entropy Coef
            self.log_alpha = torch.log(torch.ones(1) * init_value).requires_grad_(True)

            # Entropy Coef Optimizer
            self.alpha_optim = optim.Adam([self.log_alpha], lr=self.learning_rate)

## -------------------------------------------------------------------------------------------------
    def adapt(self, *p_args) -> bool:
        if not super().adapt(*p_args):
            return False

        # Adapt only when Buffer is full
        if len(self._buffer) < self.warm_up_phase:
            self.log(self.C_LOG_TYPE_I, 'Buffer is not full yet, keep collecting data!')
            return False
        
        self.log(self.C_LOG_TYPE_I, 'Buffer is full, Adapting Policy!')
    
        # Gradient Step
        for update in range(self.gradient_step):

            # Sample Data from the buffer
            sar_data = self._buffer.get_sample(self.batch_size)
            
            # Remap the data from the buffer to its own variable
            states = torch.Tensor([state.get_values() for state in sar_data["state"]])
            next_states = torch.Tensor([next_states.get_values() for next_states in sar_data["next_state"]])
            actions = torch.Tensor([action.get_sorted_values() for action in sar_data["action"]])
            rewards = torch.Tensor([reward.get_overall_reward() for reward in sar_data["reward"]])
            dones = torch.Tensor([done for done in sar_data["done"]])

            # Action by current actor for the sampled state
            action, action_log_probs, _ = self.policy.sample_action(states)

            if self.automatic_entropy_tuning:
                self.alpha = torch.exp(self.log_alpha.detach())
                ent_coef_loss = -(self.log_alpha * (action_log_probs + self.target_entropy).detach()).mean()
            else:
                self.alpha = torch.tensor(float(self.alpha)).to(self.device)

            if ent_coef_loss is not None:
                self.alpha_optim.zero_grad()
                ent_coef_loss.backward()
                self.alpha_optim.step()

            with torch.no_grad():
                # Select action according to policy
                next_state_action, next_state_log_pi, _ = self.policy.sample_action(next_states)
                if self._dist_cls == CategoricalDistribution:
                    next_state_action = next_state_action.reshape(-1,1)

                # Single Q
                next_q_values = self.critic_target(next_states, next_state_action)
                next_q_values, _ = torch.min(next_q_values, dim=1, keepdim=True)

                # add entropy term
                next_q_values = next_q_values - self.alpha.detach() * next_state_log_pi.reshape(-1, 1)

                # td error + entropy term
                target_q_values = rewards + (1 - dones) * self.gamma * next_q_values.detach().reshape(self.batch_size)

            # Single Q
            current_q_values = self.critic(states, actions)

            # Critic Loss
            critic_loss = torch.nn.functional.mse_loss(current_q_values,target_q_values.reshape(self.batch_size,1))

            # Optimze Critic Loss
            self.critic_optim.zero_grad()
            critic_loss.backward()
            self.critic_optim.step()

            # Compute Actor Loss
            # Single Q
            if self._dist_cls == CategoricalDistribution:
                action = action.reshape(-1,1)
            q_values_pi = self.critic(states, action)

            min_qf_pi, _ = torch.min(q_values_pi, dim=1, keepdim=True)
            actor_loss = (self.alpha * action_log_probs - min_qf_pi).mean()

            # Optimze Actor
            self.policy_optim.zero_grad()
            actor_loss.backward()
            self.policy_optim.step()

            if update % self.target_update_interval == 0:
                soft_update(self.critic_target, self.critic, self.tau)

        return True

## -------------------------------------------------------------------------------------------------
    def clear_buffer(self):
        self._buffer.clear()

## -------------------------------------------------------------------------------------------------
    def compute_action(self, p_state: State) -> Action:
        obs = p_state.get_values()
        
        if not isinstance(obs, torch.Tensor):
            obs = torch.Tensor(obs).reshape(1,obs.size)

        # Exploration
        if self.episode_step < self.warm_up_phase:
            action = []
            for action_dim in range(self.get_action_space().get_num_dim()):
                if len(self.get_action_space().get_dim(action_dim).get_boundaries()) == 1:
                    action.append(random.randint(0,self.get_action_space().get_dim(action_dim).get_boundaries()[0]-1))
                else:
                    action.append(random.uniform(self.get_action_space().get_dim(action_dim).get_boundaries()[0],
                                                self.get_action_space().get_dim(action_dim).get_boundaries()[1]))
            action = np.array(action)
        else:
            with torch.no_grad():
                action, _, _ = self.policy.sample_action(obs,deterministic=False)
            action = action.cpu().numpy().flatten()
        
        self.episode_step = self.episode_step + 1
        action = Action(self._id, self._action_space, action)
        return action

## -------------------------------------------------------------------------------------------------
    def _add_additional_buffer(self, p_buffer_element: SARBufferElement):
        p_buffer_element.add_value_element(self.additional_buffer_element)
        return p_buffer_element
