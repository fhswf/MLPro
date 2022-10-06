## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - A Synoptic Framework for Standardized Machine Learning Tasks
## -- Package : mlpro.rl.envmodels
## -- Module  : mlp_gridworld
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2022-10-05  0.0.0     SY       Creation
## -- 2022-10-06  1.0.0     SY       Released first version
## -------------------------------------------------------------------------------------------------

"""
Ver. 1.0.0 (2022-10-06)

This module provides Environment Model based on MLP Neural Network for grid world environment.
"""

import torch
from mlpro.rl.models import *
from mlpro.sl.pool.afct.afct_pytorch import TorchAFct
from torch.utils.data.sampler import SubsetRandomSampler
from collections import deque
from mlpro.rl.pool.envs.gridworld import *
import numpy as np
      

    
        
        
## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module
          

        
            
            
## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class GridWorldMLPModel(torch.nn.Module):
        
        
## -------------------------------------------------------------------------------------------------
    def __init__(self, n_input, n_output):
        super(GridWorldMLPModel, self).__init__()
        self.n_input = n_input
        self.n_output = n_output
        self.hidden = 128

        init_ = lambda m: init(m, torch.nn.init.orthogonal_, lambda x: torch.nn.init.
                               constant_(x, 0), np.sqrt(2))

        self.model = torch.nn.Sequential(
            init_(torch.nn.Linear(self.n_input,self.hidden)),
            torch.nn.ReLU(),
            init_(torch.nn.Linear(self.hidden,self.hidden)),
            torch.nn.ReLU(),
            init_(torch.nn.Linear(self.hidden,self.hidden)),
            torch.nn.ReLU(),
            init_(torch.nn.Linear(self.hidden,self.n_output)),
            torch.nn.ReLU()
            )
            
            
## -------------------------------------------------------------------------------------------------
    def forward(self, p_input):
        BatchSize = p_input.shape[0]
        out = self.model(p_input)   
        out = out.reshape(BatchSize, self.n_output)
        return torch.round(out)
              

            
                
                
## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class IOElement(BufferElement):
        
        
## -------------------------------------------------------------------------------------------------
    def __init__(self, p_input: torch.Tensor, p_output: torch.Tensor):
        super().__init__({"input": p_input, "output": p_output})
              

            
                
                
## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class MyOwnBuffer(Buffer, torch.utils.data.Dataset):
        
        
## -------------------------------------------------------------------------------------------------
    def __init__(self, p_size=1):
        Buffer.__init__(self, p_size=p_size)
        self._internal_counter = 0
        
        
## -------------------------------------------------------------------------------------------------
    def add_element(self, p_elem: BufferElement):
        Buffer.add_element(self, p_elem)
        self._internal_counter += 1
        
        
## -------------------------------------------------------------------------------------------------
    def get_internal_counter(self):
        return self._internal_counter
        
        
## -------------------------------------------------------------------------------------------------
    def __getitem__(self,idx):
        return self._data_buffer["input"][idx], self._data_buffer["output"][idx]
              

            
                
                
## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class GridWorldAFct(TorchAFct):
    
    C_NAME = "Grid World Adaptive Function"
    C_BUFFER_CLS = MyOwnBuffer
        
        
## -------------------------------------------------------------------------------------------------
    def _setup_model(self):
        self.net_model = GridWorldMLPModel(self._input_space.get_num_dim(),
                                           self._output_space.get_num_dim())
        self.optimizer = torch.optim.Adam(self.net_model.parameters(), lr=3e-4)
        self.criterion = torch.nn.MSELoss()
        self.train_model = True
        
        
## -------------------------------------------------------------------------------------------------
    def _adapt(self, p_input, p_output) -> bool:

        self._add_buffer(IOElement(p_input.get_values(), p_output.get_values()))

        if self._buffer.get_internal_counter() % 100 != 0:
            return False
        
        if self.train_model:
            dataset_size = len(self._buffer)
            indices = list(range(dataset_size))
            split = int(np.floor(0.3 * dataset_size))
            np.random.seed(random.randint(1,1000))
            np.random.shuffle(indices)
            train_indices, test_indices = indices[split:], indices[:split]

            train_sampler = SubsetRandomSampler(train_indices)
            test_sampler = SubsetRandomSampler(test_indices)
            trainer = torch.utils.data.DataLoader(self._buffer, batch_size=100, sampler=train_sampler)
            tester = torch.utils.data.DataLoader(self._buffer, batch_size=100, sampler=test_sampler)

            # Training
            self.net_model.train()

            for i, (In, Label) in enumerate(trainer):
                outputs = self.net_model(In.type(torch.FloatTensor))
                loss = self.criterion(outputs, Label)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            test_loss = 0

            self.net_model.eval()
            for i, (In, Label) in enumerate(tester):
                outputs = self.net_model(In.type(torch.FloatTensor))
                loss = self.criterion(outputs, Label)
                test_loss += loss.item()

            if test_loss/len(tester) < 5e-9:
                self.train_model = False

        return True
        
        
## -------------------------------------------------------------------------------------------------
    def _add_buffer(self, p_buffer_element: IOElement):
        self._buffer.add_element(p_buffer_element)
              

            
                
                
## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class MLPEnvModel(EnvModel):
    C_NAME = "MLP Env Model for Grid World"
    
    
## -------------------------------------------------------------------------------------------------
    def __init__(
        self,
        p_ada=True,
        p_logging=False,
    ):

        self.grid_world = GridWorld(p_logging=p_logging,
                                    p_action_type=GridWorld.C_ACTION_TYPE_DISC_2D)
        
        # Setup Adaptive Function
        afct_strans = AFctSTrans(
            GridWorldAFct,
            p_state_space=self.grid_world._state_space,
            p_action_space=self.grid_world._action_space,
            p_threshold=1.8,
            p_buffer_size=20000,
            p_ada=p_ada,
            p_logging=p_logging,
        )

        EnvModel.__init__(
            self,
            p_observation_space=self.grid_world._state_space,
            p_action_space=self.grid_world._action_space,
            p_latency=timedelta(seconds=0.1),
            p_afct_strans=afct_strans,
            p_afct_reward=None,
            p_afct_success=None,
            p_afct_broken=None,
            p_ada=p_ada,
            p_logging=p_logging,
        )

        self.reset()
        
        
## -------------------------------------------------------------------------------------------------
    def get_state(self) -> State:
        return self.grid_world._state

## -------------------------------------------------------------------------------------------------
    def _compute_reward(self, p_state_old: State, p_state_new: State) -> Reward:
        return self.grid_world._compute_reward(p_state_old, p_state_new)

## -------------------------------------------------------------------------------------------------
    def _compute_success(self, p_state:State) -> bool:
        return self.grid_world._compute_success(p_state)


## -------------------------------------------------------------------------------------------------
    def _compute_broken(self, p_state:State) -> bool:
        return self.grid_world._compute_broken(p_state)
