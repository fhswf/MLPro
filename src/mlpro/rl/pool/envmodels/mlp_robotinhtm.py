## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - A Synoptic Framework for Standardized Machine Learning Tasks
## -- Package : mlpro.rl.envmodels
## -- Module  : mlp_robotinhtm
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2021-12-17  0.0.0     MRD      Creation
## -- 2021-12-17  1.0.0     MRD      Released first version
## -- 2021-12-20  1.0.1     DA       Replaced 'done' by 'success'
## -- 2021-12-21  1.0.2     DA       Class MLPEnvMdel: renamed method reset() to _reset()
## -- 2022-01-02  2.0.0     MRD      Refactoring due to the changes on afct pool on
## --                                TorchAFctTrans
## -- 2022-02-25  2.0.1     SY       Refactoring due to auto generated ID in class Dimension
## -- 2022-05-22  2.0.2     MRD      Refactoring TorchAFct
## -- 2022-05-30  1.0.1     MRD      Cleaning up MLPEnvModel, now inherit directly from the
## --                                actual environment
## -------------------------------------------------------------------------------------------------

"""
Ver. 2.0.3 (2022-05-30)

This module provides Environment Model based on MLP Neural Network for
robotinhtm environment.
"""

import torch
import transformations

from mlpro.rl.models import *
from mlpro.rl.pool.envs.robotinhtm import RobotArm3D
from mlpro.rl.pool.envs.robotinhtm import RobotHTM
from mlpro.sl.pool.afct.afct_pytorch import TorchAFct

from torch.utils.data.sampler import SubsetRandomSampler
from collections import deque

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

class RobothtmAFct(TorchAFct):
    C_NAME = "Robothtm Adaptive Function"
    C_BUFFER_CLS = MyOwnBuffer

    def _setup_model(self):
        self.joint_num = self._output_space.get_num_dim() - 6
        self.net_model = RobotMLPModel(self.joint_num, 0.01)
        self.optimizer = torch.optim.Adam(self.net_model.parameters(), lr=3e-4)
        self.loss_dyn = torch.nn.MSELoss()
        self.train_model = True
        self.input_temp = None

        self.sim_env = RobotArm3D()

        joints = []

        jointType = []
        vectLinkLength = [[0, 0, 0], [0, 0, 0]]
        jointType.append("rz")
        for joint in range(self.joint_num - 1):
            vectLinkLength.append([0, 0.7, 0])
            jointType.append("rx")

        jointType.append("f")

        for x in range(len(jointType)):
            vectorLink = dict(x=vectLinkLength[x][0], y=vectLinkLength[x][1], z=vectLinkLength[x][2])
            joint = dict(
                Joint_name="Joint %d" % x,
                Joint_type=jointType[x],
                Vector_link_length=vectorLink,
            )
            joints.append(joint)

        for robo in joints:
            self.sim_env.add_link_joint(
                lvector=torch.Tensor(
                    [
                        [
                            robo["Vector_link_length"]["x"],
                            robo["Vector_link_length"]["y"],
                            robo["Vector_link_length"]["z"],
                        ]
                    ]
                ),
                jointAxis=robo["Joint_type"],
                thetaInit=torch.Tensor([np.radians(0)]),
            )

        self.sim_env.update_joint_coords()

    def _input_preproc(self, p_input: torch.Tensor) -> torch.Tensor:
        input = torch.cat([p_input[0][6+self.joint_num:], p_input[0][6:6+self.joint_num]])
        input = input.reshape(1,self.joint_num*2)
        self.input_temp = p_input[0][:3].reshape(1,3)
        
        return input

    def _output_postproc(self, p_output: torch.Tensor) -> torch.Tensor:
        angles = torch.Tensor([])
        thets = torch.zeros(3)
        for idx in range(self.joint_num):
            angle = torch.Tensor(transformations.euler_from_quaternion(p_output[-1][idx][3:].detach().numpy(), axes="rxyz")) - thets
            thets = torch.Tensor(transformations.euler_from_quaternion(p_output[-1][idx][3:].detach().numpy(), axes="rxyz"))
            angles = torch.cat([angles, torch.norm(angle).reshape(1, 1)], dim=1)

        output = torch.cat([self.input_temp, p_output[-1][-1][:3].reshape(1,3)], dim=1)
        output = torch.cat([output, angles], dim=1)

        return output
    
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
            self.net_model.train()

            for i, (In, Label) in enumerate(trainer):
                outputs = self.net_model(In)
                loss = self.loss_dyn(outputs, Label)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            test_loss = 0

            self.net_model.eval()
            for i, (In, Label) in enumerate(tester):
                outputs = self.net_model(In)
                loss = self.loss_dyn(outputs, Label)
                test_loss += loss.item()

            if test_loss/len(tester) < 5e-9:
                self.train_model = False

        return True

    def _add_buffer(self, p_buffer_element: IOElement):
        self._buffer.add_element(p_buffer_element)

class MLPEnvModel(RobotHTM, EnvModel):
    C_NAME = "MLP Env Model"

    def __init__(
        self,
        p_num_joints=4,
        p_target_mode="Random",
        p_ada=True,
        p_logging=False,
    ):

        RobotHTM.__init__(self, p_num_joints=p_num_joints, p_target_mode=p_target_mode)
        
        # Setup Adaptive Function
        # HTM Function Here
        afct_strans = AFctSTrans(
            RobothtmAFct,
            p_state_space=self._state_space,
            p_action_space=self._action_space,
            p_threshold=1.8,
            p_buffer_size=20000,
            p_ada=p_ada,
            p_logging=p_logging,
        )

        EnvModel.__init__(
            self,
            p_observation_space=self._state_space,
            p_action_space=self._action_space,
            p_latency=timedelta(seconds=self.dt),
            p_afct_strans=afct_strans,
            p_afct_reward=None,
            p_afct_success=None,
            p_afct_broken=None,
            p_ada=p_ada,
            p_logging=p_logging,
        )

        self.reset()

