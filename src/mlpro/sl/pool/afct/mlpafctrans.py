## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - A Synoptic Framework for Standardized Machine Learning Tasks
## -- Package : mlpro.sl.afct
## -- Module  : mlpafctrans
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2021-12-17  0.0.0     MRD       Creation
## -- 2021-12-17  1.0.0     MRD       Released first version
## -------------------------------------------------------------------------------------------------

"""
Ver. 1.0.0 (2021-12-17)

This module provides Adaptive Functions for state transition based
on robotinhtm environment.
"""

import torch
from torch.utils.data.sampler import SubsetRandomSampler
import transformations

from collections import deque

from mlpro.rl.models import *

def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module

class RobotMLPModel(torch.nn.Module):
    def __init__(self, n_joint, timeStep):
        super(RobotMLPModel, self).__init__()
        self.n_joint = n_joint
        self.timeStep = timeStep
        self.hidden = 128

        init_ = lambda m: init(m, torch.nn.init.orthogonal_, lambda x: torch.nn.init.
                               constant_(x, 0), np.sqrt(2))

        self.model1 = torch.nn.Sequential(
            init_(torch.nn.Linear(self.n_joint,self.hidden)),
            torch.nn.Tanh(),
            init_(torch.nn.Linear(self.hidden,self.hidden)),
            torch.nn.Tanh(),
            init_(torch.nn.Linear(self.hidden,self.hidden)),
            torch.nn.Tanh(),
            init_(torch.nn.Linear(self.hidden,7*(self.n_joint+1))),
            torch.nn.Tanh()
            )
        
    def forward(self, I):
        BatchSize=I.shape[0]
        newI = I.reshape(BatchSize,2,self.n_joint) * torch.cat([torch.Tensor([self.timeStep]).repeat(1,self.n_joint), torch.ones(1,self.n_joint)])
        newI = torch.sum(newI,dim=1)
        out2 = self.model1(newI)   
        out2 = out2.reshape(BatchSize,self.n_joint+1,7)
        return out2


# Input Output Buffer
class IOElement(BufferElement):
    def __init__(self, p_input: torch.Tensor, p_output: torch.Tensor):

        super().__init__({"input": p_input, "output": p_output})


# Buffer
class MyOwnBuffer(Buffer, torch.utils.data.Dataset):
    def __init__(self, p_size=1):
        Buffer.__init__(self, p_size=p_size)
        self._internal_counter = 0

    def add_element(self, p_elem: BufferElement):
        Buffer.add_element(self, p_elem)
        self._internal_counter += 1

    def get_internal_counter(self):
        return self._internal_counter

    def __getitem__(self,idx):
        return self._data_buffer["input"][idx], self._data_buffer["output"][idx]


class MLPAFctTrans(AdaptiveFunction):
    C_NAME = "HTM Adaptive Function"
    C_BUFFER_CLS = MyOwnBuffer

    def __init__(
        self,
        p_input_space: MSpace,
        p_output_space: MSpace,
        p_output_elem_cls=Element,
        p_threshold=0,
        p_buffer_size=0,
        p_ada=True,
        p_logging=Log.C_LOG_ALL,
        **p_par
    ):

        super().__init__(
            p_input_space=p_input_space,
            p_output_space=p_output_space,
            p_output_elem_cls=p_output_elem_cls,
            p_threshold=p_threshold,
            p_buffer_size=p_buffer_size,
            p_ada=p_ada,
            p_logging=p_logging,
            **p_par
        )

        self.joint_num = p_output_space.get_num_dim() - 6
        self.mlp_model = RobotMLPModel(self.joint_num, 0.01)
        self.optimizer = torch.optim.Adam(self.mlp_model.parameters(), lr=3e-4)
        self.loss_dyn = torch.nn.MSELoss()
        self.sim_env = p_par["p_sim_env"]
        self.train_model = True

    def _map(self, p_input: Element, p_output: Element):
        # Prediction
        # Input [Tx, Ty, Tz, Px, Py, Pz, J1, J2, J3, J4, A1, A2, A3, A4]
        # Model Input [A1, A2, A3, A4, J1, J2, J3, J4]
        model_input = deque(p_input.get_values()[6:])
        model_input.rotate(self.joint_num)
        model_input = torch.Tensor([list(model_input)])
        model_output = self.mlp_model(model_input)

        # Model Output [Px, Py, Pz, Rw, Rx, Ry, Rz]
        # Output [Tx, Ty, Tz, Px, Py, Pz, J1, J2, J3, J4]

        # Convert [HTM1, HTM2, HTM3, HTM4] to [J1, J2, J3, J4]
        angles = torch.Tensor([])
        thets = torch.zeros(3)
        for idx in range(self.joint_num):
            angle = torch.Tensor(transformations.euler_from_quaternion(model_output[-1][idx][3:].detach().numpy(), axes="rxyz")) - thets
            thets = torch.Tensor(transformations.euler_from_quaternion(model_output[-1][idx][3:].detach().numpy(), axes="rxyz"))
            angles = torch.cat([angles, torch.norm(angle).reshape(1, 1)], dim=1)

        # Combine Output
        output = p_input.get_values()[:3].copy()
        output.extend(model_output[-1][-1][:3].cpu().flatten().tolist())
        output.extend(angles.cpu().flatten().tolist())
        p_output.set_values(output)

    def _adapt(self, p_input: Element, p_output: Element) -> bool:
        model_input = deque(p_input.get_values()[6:])
        model_input.rotate(self.joint_num)
        model_input = torch.Tensor([list(model_input)])

        self.sim_env.set_theta(torch.Tensor([p_output.get_values()[6 : 6 + self.joint_num]]))
        self.sim_env.update_joint_coords()

        model_output = self.sim_env.convert_to_quaternion().reshape(1,self.joint_num+1,7)

        self._add_buffer(IOElement(model_input, model_output))

        if self._buffer.get_internal_counter() % 100 != 0:
            return False

        # Divide Test and Train
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
            self.mlp_model.train()

            for i, (In, Label) in enumerate(trainer):
                outputs = self.mlp_model(In)
                loss = self.loss_dyn(outputs, Label)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            test_loss = 0

            self.mlp_model.eval()
            for i, (In, Label) in enumerate(tester):
                outputs = self.mlp_model(In)
                loss = self.loss_dyn(outputs, Label)
                test_loss += loss.item()

            print(test_loss/len(tester))
            if test_loss/len(tester) < 5e-9:
                self.train_model = False

        return True

    def _add_buffer(self, p_buffer_element: IOElement):
        self._buffer.add_element(p_buffer_element)

