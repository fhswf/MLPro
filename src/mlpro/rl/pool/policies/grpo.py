## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - The integrative middleware framework for standardized machine learning
## -- Package : mlpro.rl.pool.policies
## -- Module  : grpo.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2025-04-02  0.0.0     SY       Creation
## -- 2025-04-10  1.0.0     SY       Release of first version
## -------------------------------------------------------------------------------------------------

"""
Ver. 1.0.0 (2025-04-10)

This module implements a minimal version of Group Relative Policy Optimization (GRPO) for continuous
action spaces, as described in the paper published on arXiv:2402.03300.
"""


from mlpro.rl import *
from mlpro.bf.data import *
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
import numpy as np
import copy
import random
         


        
        
## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class MinGRPOPolicyNetwork(nn.Module):
    """
    This is a default implementation of the minimal GRPO policy network designed for continuous
    action spaces. You also have the option to define your own network, provided it includes a
    forward function that accepts inputs and produces outputs of the same type as those in this
    class.

    Parameters
    ----------
    p_state_space : MSpace
        MLPro-compatible definition of the observation (state) space.
    p_action_space : MSpace
        MLPro-compatible definition of the continuous action space.
    p_hidden_layers : list
        A list specifying the number of neurons in each hidden layer. For example, [128, 128]
        indicates two hidden layers, each with 128 neurons. Default value: [128, 128].
    p_seed : int
        Seeding (optional). Default value: None.
    """


## -------------------------------------------------------------------------------------------------
    def __init__(self, p_state_space:MSpace, p_action_space:MSpace, p_hidden_layers:list=[128, 128], p_seed:int=None):
        
        super().__init__()
        self.state_space = p_state_space
        self.action_space = p_action_space
        
        if p_seed is not None:
            self.set_seed(p_seed)
        
        layers              = []
        input_dim           = self.state_space.get_num_dim()
        for hidden_dim in p_hidden_layers:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            input_dim = hidden_dim
            
        self.fc             = nn.Sequential(*layers)
        self.actor          = nn.Linear(input_dim, self.action_space.get_num_dim())
        self.actor_logstd   = nn.Parameter(torch.zeros(1, self.action_space.get_num_dim()))
        self.critic         = nn.Linear(input_dim, 1)
        
        for layer in self.fc:
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))
                nn.init.constant_(layer.bias, 0.0)
        
        nn.init.orthogonal_(self.actor.weight, gain=0.01)
        nn.init.constant_(self.actor.bias, 0.0)
        nn.init.orthogonal_(self.critic.weight, gain=1.0)
        nn.init.constant_(self.critic.bias, 0.0)
        nn.init.constant_(self.actor_logstd, -1.0) 
        
        self.low_bound      = torch.tensor([act.get_boundaries()[0] for act in self.action_space.get_dims()])
        self.high_bound     = torch.tensor([act.get_boundaries()[1] for act in self.action_space.get_dims()])


## -------------------------------------------------------------------------------------------------
    def set_seed(self, seed):
        """
        Sets the random seed for reproducibility.

        Parameters
        ----------
        seed : int
            The seed to be set for random number generation.
        """
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)


## -------------------------------------------------------------------------------------------------
    def forward(self, state:torch.Tensor):
        """
        Performs a forward pass through the policy network.

        The network processes the input state through fully connected layers to compute:
            - The mean and log standard deviation of a Gaussian distribution over actions.
            - The estimated value of the current state (for the critic).

        Parameters
        ----------
        state : torch.Tensor
            A batch of input states with shape (batch_size, state_dim), where each row represents a
            single state in the environment.

        Returns
        -------
        action_mean : torch.Tensor of shape (batch_size, action_dim)
            The mean of the Gaussian distribution for each action dimension.
        action_logstd : torch.Tensor of shape (batch_size, action_dim)
            The log standard deviation of the Gaussian distribution. It is expanded from a
            learnable parameter to match the batch size.
        state_value : torch.Tensor of shape (batch_size, 1)
            The estimated value of each input state.
        """
        
        state           = self.fc(state)
        action_mean     = self.actor(state)
        action_logstd   = self.actor_logstd.expand_as(action_mean)
        state_value     = self.critic(state)

        return (action_mean, action_logstd), state_value                 


                
                
## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class MinGRPO(Policy):
    """
    Minimal implementation of Group Relative Policy Optimization (GRPO) for continuous action
    spaces, based on the algorithm introduced in the paper:
        
    Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, et al. "DeepSeekMath: Pushing the 
    Limits of Mathematical Reasoning in Open Language Models." arXiv:2402.03300 (2024).
    
    Minimal GRPO is a variant of PPO that introduces group-based policy updates by partitioning
    experiences into high-reward and low-reward groups and applying distinct weighting schemes to
    each group during optimization.

    Parameters
    ----------
    p_network : nn.Module
        A PyTorch neural network implementing the policy and value functions. Must return both
        (action_mean, action_logstd) and state value during forward pass.
    p_optimizer : torch.optim.Optimizer
        Optimizer used to update the policy network's parameters.
    p_observation_space : MSpace
        MLPro-compatible definition of the observation (state) space.
    p_action_space : MSpace
        MLPro-compatible definition of the continuous action space.
    p_buffer_size : int
        Number of transitions to store before each update, if termination is not reached.
    p_ada : bool, optional
        Whether adaptive training is enabled. Default is True.
    p_visualize : bool, optional
        Enable or disable visualization. Default is False.
    p_logging : int, optional
        Logging level as defined in MLPro’s Log class. Default is Log.C_LOG_ALL.
    p_gamma : float, optional
        Discount factor for future rewards (default: 0.99).
    p_lam : float, optional
        Lambda for Generalized Advantage Estimation (GAE) (default: 0.95).
    p_epsilon : float, optional
        Small constant to avoid division by zero in advantage normalization (default: 1e-8).
    p_max_norm : float, optional
        Maximum norm for gradient clipping (default: 0.5).
    p_clip_eps : float, optional
        Clipping range for policy ratio in GRPO (default: 0.2).
    p_group_ratio : float, optional
        Ratio of transitions considered as the "high" reward group (default: 0.5).
    p_low_weight : float, optional
        Weight for the loss from the "low" reward group (default: 2.0).
    p_value_loss_weight : float, optional
        Weight for the value function loss component (default: 0.5).
    p_entropy_weight : float, optional
        Entropy bonus to encourage exploration (default: 0.01).
    p_kl_weight : float, optional
        Weight for KL divergence between new and old policies (default: 0.01).
    p_minibatch_size : int, optional
        Number of samples per gradient update. Larger values increase stability but reduce learning
        speed (default: 64).
    p_n_epochs : int, optional
        Number of passes through the buffer data during optimization. Higher values enable better
        data utilization but risk overfitting (default: 10).
    p_logstd_min : float, optional
        Minimum allowed value for log standard deviation of action distribution. Prevents numerical
        underflow in policy updates (default: -20.0).
    p_logstd_max : float, optional
        Maximum allowed value for log standard deviation of action distribution. Prevents numerical
        overflow in policy updates (default: 2.0).
    p_log_epsilon : float, optional
        Small constant to stabilize log-probability calculations for bounded actions. Used in
        tanh-transformed action probability calculations (default: 1e-6).
    p_std_epsilon : float, optional
        Small constant to avoid division by zero in reward/advantage normalization (default: 1e-8).
    p_ratio_min : float, optional
        Minimum clipping value for importance sampling ratio. Controls conservative policy updates
        (default: 0.1).
    p_ratio_max : float, optional
        Maximum clipping value for importance sampling ratio. Limits maximum policy update steps
        (default: 10.0).
    """

    C_NAME          = "Minimal GRPO"
    C_BUFFER_CLS    = BufferRnd


## -------------------------------------------------------------------------------------------------
    def __init__(
            self,
            p_network:nn.Module,
            p_optimizer:torch.optim.Optimizer,
            p_observation_space:MSpace, 
            p_action_space:MSpace, 
            p_buffer_size:int,
            p_ada:bool=True, 
            p_visualize:bool=False,
            p_logging=Log.C_LOG_ALL,
            p_gamma:float=None,
            p_lam:float=None,
            p_epsilon:float=None,
            p_max_norm:float=None,
            p_clip_eps:float=None,
            p_group_ratio:float=None,
            p_low_weight:float=None,
            p_value_loss_weight:float=None,
            p_entropy_weight:float=None,
            p_kl_weight:float=None,
            p_minibatch_size:int=None,
            p_n_epochs:int=None,
            p_logstd_min:float=None,
            p_logstd_max:float=None,
            p_log_epsilon:float=None,
            p_std_epsilon:float=None,
            p_ratio_min:float=None,
            p_ratio_max:float=None
            ):

        super().__init__ ( p_observation_space=p_observation_space, 
                           p_action_space=p_action_space, 
                           p_buffer_size=p_buffer_size, 
                           p_ada=p_ada, 
                           p_visualize=p_visualize, 
                           p_logging=p_logging )
        
        self.C_SCIREF_TYPE      = self.C_SCIREF_TYPE_ARTICLE
        self.C_SCIREF_AUTHOR    = "Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, et al."
        self.C_SCIREF_TITLE     = "DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models"
        self.C_SCIREF_JOURNAL   = "arXiv preprint arXiv:2402.03300"
        self.C_SCIREF_YEAR      = "2024"
        
        if not isinstance(p_network, nn.Module):
            raise ParamError("p_network is not inherited from nn.Module class.")
        if not isinstance(p_optimizer, optim.Optimizer):
            raise ParamError("p_optimizer is not inherited from torch.optim.Optimizer class.")
        self._network           = p_network
        self._optimizer         = p_optimizer

        self._state_space       = p_observation_space
        self._action_space      = p_action_space
        self._log_probs_elem    = {}
        self._values_elem       = {}
        self._old_network       = copy.deepcopy(self._network)
        self._old_network.eval()
        
        self._hyperparam_space  = HyperParamSpace()
        self._hyperparam_tuple  = None
        self._init_hyperparam()
        
        ids_                    = self._hyperparam_tuple.get_dim_ids()
        if p_gamma is not None:
            self._hyperparam_tuple.set_value(ids_[0], p_gamma)
        if p_lam is not None:
            self._hyperparam_tuple.set_value(ids_[1], p_lam)
        if p_epsilon is not None:
            self._hyperparam_tuple.set_value(ids_[2], p_epsilon)
        if p_max_norm is not None:
            self._hyperparam_tuple.set_value(ids_[3], p_max_norm)
        if p_clip_eps is not None:
            self._hyperparam_tuple.set_value(ids_[4], p_clip_eps)
        if p_group_ratio is not None:
            self._hyperparam_tuple.set_value(ids_[5], p_group_ratio)
        if p_low_weight is not None:
            self._hyperparam_tuple.set_value(ids_[6], p_low_weight)
        if p_value_loss_weight is not None:
            self._hyperparam_tuple.set_value(ids_[7], p_value_loss_weight)
        if p_entropy_weight is not None:
            self._hyperparam_tuple.set_value(ids_[8], p_entropy_weight)
        if p_kl_weight is not None:
            self._hyperparam_tuple.set_value(ids_[9], p_kl_weight)
        if p_minibatch_size is not None:
            self._hyperparam_tuple.set_value(ids_[10], p_minibatch_size)
        if p_n_epochs is not None:
            self._hyperparam_tuple.set_value(ids_[11], p_n_epochs)
        if p_logstd_min is not None:
            self._hyperparam_tuple.set_value(ids_[12], p_logstd_min)
        if p_logstd_max is not None:
            self._hyperparam_tuple.set_value(ids_[13], p_logstd_max)
        if p_log_epsilon is not None:
            self._hyperparam_tuple.set_value(ids_[14], p_log_epsilon)
        if p_std_epsilon is not None:
            self._hyperparam_tuple.set_value(ids_[15], p_std_epsilon)
        if p_ratio_min is not None:
            self._hyperparam_tuple.set_value(ids_[16], p_ratio_min)
        if p_ratio_max is not None:
            self._hyperparam_tuple.set_value(ids_[17], p_ratio_max)
        self._hp_ids            = self.get_hyperparam().get_dim_ids()
        
        
## -------------------------------------------------------------------------------------------------
    def _init_hyperparam(self):
        """
        Initializes the hyperparameter space and sets default values for the GRPO algorithm.
        """
        
        self._hyperparam_space.add_dim(HyperParam('gamma','R'))
        self._hyperparam_space.add_dim(HyperParam('lam','R'))
        self._hyperparam_space.add_dim(HyperParam('epsilon','R'))
        self._hyperparam_space.add_dim(HyperParam('max_norm','R'))
        self._hyperparam_space.add_dim(HyperParam('clip_eps','R'))
        self._hyperparam_space.add_dim(HyperParam('group_ratio','R'))
        self._hyperparam_space.add_dim(HyperParam('low_weight','R'))
        self._hyperparam_space.add_dim(HyperParam('value_loss_weight','R'))
        self._hyperparam_space.add_dim(HyperParam('entropy_weight','R'))
        self._hyperparam_space.add_dim(HyperParam('kl_weight','R'))
        self._hyperparam_space.add_dim(HyperParam('minibatch_size','Z'))
        self._hyperparam_space.add_dim(HyperParam('n_epochs ','Z'))
        self._hyperparam_space.add_dim(HyperParam('logstd_min','R'))
        self._hyperparam_space.add_dim(HyperParam('logstd_max','R'))
        self._hyperparam_space.add_dim(HyperParam('log_epsilon','R'))
        self._hyperparam_space.add_dim(HyperParam('std_epsilon','R'))
        self._hyperparam_space.add_dim(HyperParam('ratio_min','R'))
        self._hyperparam_space.add_dim(HyperParam('ratio_max','R'))
        self._hyperparam_tuple = HyperParamTuple(self._hyperparam_space)
        
        ids_ = self._hyperparam_tuple.get_dim_ids()
        self._hyperparam_tuple.set_value(ids_[0], 0.99)
        self._hyperparam_tuple.set_value(ids_[1], 0.95)
        self._hyperparam_tuple.set_value(ids_[2], 1e-8)
        self._hyperparam_tuple.set_value(ids_[3], 0.5)
        self._hyperparam_tuple.set_value(ids_[4], 0.2)
        self._hyperparam_tuple.set_value(ids_[5], 0.5)
        self._hyperparam_tuple.set_value(ids_[6], 2.0)
        self._hyperparam_tuple.set_value(ids_[7], 0.5)
        self._hyperparam_tuple.set_value(ids_[8], 0.01)
        self._hyperparam_tuple.set_value(ids_[9], 0.01)
        self._hyperparam_tuple.set_value(ids_[10], 64)
        self._hyperparam_tuple.set_value(ids_[11], 10)
        self._hyperparam_tuple.set_value(ids_[12], -20.0)
        self._hyperparam_tuple.set_value(ids_[13], 2.0)
        self._hyperparam_tuple.set_value(ids_[14], 1e-6)
        self._hyperparam_tuple.set_value(ids_[15], 1e-8)
        self._hyperparam_tuple.set_value(ids_[16], 0.1)
        self._hyperparam_tuple.set_value(ids_[17], 10.0)
        

## -------------------------------------------------------------------------------------------------
    def add_buffer(self, p_buffer_element:SARSElement):
        """
        Adds a transition element (SARSElement) to the internal replay buffer after enriching it
        with additional policy-specific metadata.
    
        Parameters
        ----------
        p_buffer_element : SARSElement
            A transition tuple containing state, action, reward, and next state.
        """
        
        buffer_element = self._add_additional_buffer(p_buffer_element)
        self._buffer.add_element(buffer_element)


## -------------------------------------------------------------------------------------------------
    def _add_additional_buffer(self, p_buffer_element:SARSElement) -> SARSElement:
        """
        Adds extra policy-specific information to a buffer element before storing.
    
        Specifically, this function attaches the log-probability of the taken action and the
        estimated state value to the transition for use in advantage estimation and policy gradient
        calculations.
    
        Parameters
        ----------
        p_buffer_element : SARSElement
            A transition element containing the current state, action, reward, and next state.
    
        Returns
        -------
        SARSElement
            The modified buffer element including additional GRPO-specific metadata.
        """
        
        p_buffer_element.add_value_element(self._log_probs_elem)
        p_buffer_element.add_value_element(self._values_elem)
        
        return p_buffer_element


## -------------------------------------------------------------------------------------------------
    def clear_buffer(self):
        """
        Clears the internal replay buffer.
        """
        
        self._buffer.clear()


## -------------------------------------------------------------------------------------------------
    def compute_action(self, p_state:State) -> Action:
        """
        Computes an action for a given environment state using the current policy.
    
        The method passes the state through the policy network to sample a stochastic action from a
        Gaussian distribution. It also logs the log-probabilities and critic value for training
        purposes.
    
        Parameters
        ----------
        p_state : State
            The current state from the environment.
    
        Returns
        -------
        Action
            A sampled action encapsulated in MLPro's Action object.
        """
        
        state           = torch.tensor(p_state.get_values(), dtype=torch.float32).unsqueeze(0)
        (action_mean, action_logstd), state_value = self._network(state)
        action_logstd   = torch.clamp(
            action_logstd,
            min=self.get_hyperparam().get_value(self._hp_ids[12]),
            max=self.get_hyperparam().get_value(self._hp_ids[13])
            )
        dist            = Normal(action_mean, action_logstd.exp())
        raw_action      = dist.rsample()
        action          = torch.tanh(raw_action)
        action_low      = self._network.low_bound
        action_high     = self._network.high_bound
        scaled_action   = action_low+(action+1)*(action_high-action_low)/2
        
        log_probs       = dist.log_prob(raw_action).sum(dim=-1)
        log_probs       -= torch.log(1-action.pow(2)+self.get_hyperparam().get_value(self._hp_ids[14])).sum(dim=-1)
        
        self._log_probs_elem    = dict(log_prob=log_probs.item())
        self._values_elem       = dict(value=state_value.item())
        
        my_action_values        = []
        for d in range(self._action_space.get_num_dim()):
            my_action_values.append(scaled_action[0][d].item()) 

        return Action(self._id, self._action_space, my_action_values)


## -------------------------------------------------------------------------------------------------
    def _adapt(self, **p_kwargs) -> bool:
        """
        Performs a policy update when enough transitions have been collected.
    
        Transitions are accumulated in a buffer until it is full. Then, the method computes
        advantages using GAE, normalizes them, and performs a gradient-based update using the GRPO
        loss function.
    
        Parameters
        ----------
        p_kwargs : dict
           A dictionary that must contain the key `'p_sars_elem'` with a SARSElement value.
    
        Returns
        -------
        bool
            True if a policy update was performed, False otherwise.
        """
        
        self.add_buffer(p_kwargs["p_sars_elem"])
        new_state               = p_kwargs["p_sars_elem"].get_data()["state_new"]
                
        if self._buffer.is_full() or new_state.get_terminal():
            minibatch_size = int(self.get_hyperparam().get_value(self._hp_ids[10]))
            if self._buffer.__len__() < minibatch_size:
                pass
            else:
                for _ in range(int(self.get_hyperparam().get_value(self._hp_ids[11]))):
                    buffer_idx          = self._buffer._gen_sample_ind(minibatch_size)
                    buffer_data         = self._buffer._extract_rows(buffer_idx)
                    
                    if buffer_data["reward"][0].type == 0:
                        raw_rewards     = [r.get_overall_reward() for r in buffer_data["reward"]]
                    else:
                        raw_rewards     = [sum(r.rewards) for r in buffer_data["reward"]]
                    rewards_tensor      = torch.FloatTensor(raw_rewards)
                    values_tensor       = torch.FloatTensor(buffer_data["value"]+[0.0])
                    returns             = self._compute_gae(rewards_tensor, values_tensor)
                    advantages          = returns-values_tensor[:-1]
                    epsilon             = self.get_hyperparam().get_value(self._hp_ids[2])
                    advantages          = (advantages-advantages.mean())/(advantages.std()+epsilon)
                    
                    states_np           = np.array([st.get_values() for st in buffer_data["state"]])
                    states_tensor       = torch.from_numpy(states_np).float()
                    actions_np          = np.array([act.get_sorted_values() for act in buffer_data["action"]])
                    actions_tensor      = torch.from_numpy(actions_np).float()
                    
                    old_network         = copy.deepcopy(self._network)
                    old_network.eval()
                    
                    self._optimizer.zero_grad()
                    loss                = self._grpo_loss(states_tensor, actions_tensor, returns, advantages)
                    loss.backward()
                    max_norm            = self.get_hyperparam().get_value(self._hp_ids[3])
                    torch.nn.utils.clip_grad_norm_(self._network.parameters(), max_norm=max_norm, norm_type=2.0)
                    self._optimizer.step()
                        
                    self._old_network   = copy.deepcopy(old_network)
                    self._old_network.eval()
            
            if self._buffer.is_full():
                self._buffer.clear()
            return True
        else:
            return False


## -------------------------------------------------------------------------------------------------
    def _compute_gae(self, rewards:torch.Tensor, values:torch.Tensor) -> torch.Tensor:
        """
        Computes returns using GAE. This function estimates advantage values that balance bias and
        variance by blending n-step returns with bootstrapped values.
    
        Parameters
        ----------
        rewards : torch.Tensor
            Tensor of rewards collected during rollout.
        values : torch.Tensor
            Tensor of value estimates for each state (with one extra value for bootstrap).
    
        Returns
        -------
        torch.Tensor
            Tensor of return values used for policy gradient and value function updates.
        """
        
        rewards     = (rewards-rewards.mean())/(rewards.std()+self.get_hyperparam().get_value(self._hp_ids[15]))
        T           = len(rewards)
        advantages  = torch.zeros_like(rewards)
        gae         = 0.0
        gamma       = self.get_hyperparam().get_value(self._hp_ids[0])
        lam         = self.get_hyperparam().get_value(self._hp_ids[1])
        
        for t in reversed(range(T)):
            delta           = rewards[t]+gamma*values[t+1]-values[t]
            gae             = delta+gamma*lam*gae
            advantages[t]   = gae
        
        return advantages+values[:-1]      


## -------------------------------------------------------------------------------------------------
    def _grpo_loss(self, states:torch.Tensor, actions:torch.Tensor, returns:torch.Tensor,
                   advantages:torch.Tensor) -> float:
        """
        Computes the GRPO loss.
    
        The loss consists of multiple components:
            - PPO-style clipped policy loss for high and low reward groups.
            - Weighted value function loss.
            - Entropy bonus to encourage exploration.
            - KL divergence regularization for policy stability.
    
        Parameters
        ----------
        states : torch.Tensor
            Batch of observed states.
        actions : torch.Tensor
            Batch of actions taken in those states.
        returns : torch.Tensor
            Estimated returns using GAE.
        advantages : torch.Tensor
            Normalized advantage estimates.
    
        Returns
        -------
        float
            The total GRPO loss to be minimized.
        """
        
        clip_eps            = self.get_hyperparam().get_value(self._hp_ids[4])
        group_ratio         = self.get_hyperparam().get_value(self._hp_ids[5])
        low_weight          = self.get_hyperparam().get_value(self._hp_ids[6])
        val_weight          = self.get_hyperparam().get_value(self._hp_ids[7])
        ent_weight          = self.get_hyperparam().get_value(self._hp_ids[8])
        kl_weight           = self.get_hyperparam().get_value(self._hp_ids[9])
        
        # 1. Evaluate both policies on the same states
        (new_mean, new_logstd), new_values = self._network(states)
        (old_mean, old_logstd), old_values = self._old_network(states)
        
        new_dist            = Normal(new_mean, new_logstd.exp())
        old_dist            = Normal(old_mean, old_logstd.exp())
    
        new_log_probs       = new_dist.log_prob(actions).sum(dim=-1)
        old_log_probs       = old_dist.log_prob(actions).sum(dim=-1)
        entropy             = new_dist.entropy().mean()
        
        # 2. Grouping by reward difference (new vs old)
        with torch.no_grad():
            delta_rewards   = (new_values-old_values).squeeze()
            sorted_indices  = torch.argsort(delta_rewards, descending=True)
            split_idx       = int(len(states)*group_ratio)
            high_group      = sorted_indices[:split_idx]
            low_group       = sorted_indices[split_idx:]
            
        # 3. Value loss
        value_loss          = (returns-new_values).pow(2).mean()
            
        # 4. Group-relative policy loss
        ratio_min           = self.get_hyperparam().get_value(self._hp_ids[16])
        ratio_max           = self.get_hyperparam().get_value(self._hp_ids[17])
        
        advantages          = (advantages-advantages.mean())/(advantages.std()+self.get_hyperparam().get_value(self._hp_ids[15]))
        group_adv           = advantages[high_group]
        ratio               = torch.exp((new_log_probs[high_group]-old_log_probs[high_group]))
        ratio               = torch.clamp(ratio, ratio_min, ratio_max)
        try:
            clip_adv        = torch.clamp(ratio, 1-clip_eps, 1+clip_eps)*group_adv
            high_loss       = -torch.min(ratio*group_adv, clip_adv).mean()
        except:
            clip_adv        = torch.clamp(ratio, 1-clip_eps, 1+clip_eps)*group_adv[:,0]
            high_loss       = -torch.min(ratio*group_adv[:,0], clip_adv).mean()
        
        group_adv           = advantages[low_group]
        ratio               = torch.exp((new_log_probs[low_group]-old_log_probs[low_group]))
        ratio               = torch.clamp(ratio, ratio_min, ratio_max)
        try:
            clip_adv        = torch.clamp(ratio, 1-clip_eps, 1+clip_eps)*group_adv
            low_loss        = -torch.min(ratio*group_adv, clip_adv).mean()
        except:
            clip_adv        = torch.clamp(ratio, 1-clip_eps, 1+clip_eps)*group_adv[:,0]
            low_loss        = -torch.min(ratio*group_adv[:,0], clip_adv).mean()
        
        # 5. Total loss and KL divergence loss
        total_loss          = (
            high_loss+low_weight*low_loss+
            val_weight*value_loss-
            ent_weight*entropy
            )
        
        kl_div              = torch.distributions.kl_divergence(old_dist, new_dist).mean()
        total_loss          += kl_weight*kl_div
        
        return total_loss