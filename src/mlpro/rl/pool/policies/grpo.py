## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - The integrative middleware framework for standardized machine learning
## -- Package : mlpro.rl.pool.policies
## -- Module  : grpo.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2025-04-02  0.0.0     SY       Creation
## -- 2025-04-08  1.0.0     SY       Release of first version
## -------------------------------------------------------------------------------------------------

"""
Ver. 1.0.0 (2025-04-08)

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
    state_dim : int
        The dimension of the state space.
    action_dim : int
        The dimension of the action space.
    hidden_layers : list
        A list specifying the number of neurons in each hidden layer. For example, [128, 128]
        indicates two hidden layers, each with 128 neurons. Default value: [128, 128].
    """


## -------------------------------------------------------------------------------------------------
    def __init__(self, state_dim:int, action_dim:int, hidden_layers=[128, 128]):
        super().__init__()
        
        layers              = []
        input_dim           = state_dim
        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            input_dim = hidden_dim
            
        self.fc             = nn.Sequential(*layers)
        self.actor          = nn.Linear(input_dim, action_dim)
        self.actor_logstd   = nn.Parameter(torch.zeros(1, action_dim))
        self.critic         = nn.Linear(input_dim, 1)


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
        action_mean     = torch.tanh(self.actor(state))
        action_logstd   = self.actor_logstd.expand_as(action_mean)
        state_value     = self.critic(state)
        
        return (action_mean, action_logstd), state_value