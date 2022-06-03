## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - A Synoptic Framework for Standardized Machine Learning Tasks
## -- Package : mlpro.rl.envmodels
## -- Module  : mlp_robotinhtm
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2022-05-20  0.0.0     MRD      Creation
## -- 2022-05-22  1.0.0     MRD      Release first version
## -- 2022-05-30  1.0.1     MRD      Cleaning up HTMEnvModel, now inherit directly from the
## --                                actual environment
## -------------------------------------------------------------------------------------------------

"""
Ver. 1.0.1 (2022-05-30)

This module provides Environment Model based on Homogeneous Transformations Matrix 
Neural Network for robotinhtm environment.
"""

import torch
import transformations

from mlpro.rl.models import *
from mlpro.rl.pool.envs.robotinhtm import RobotArm3D
from mlpro.rl.pool.envs.robotinhtm import RobotHTM
from mlpro.sl.pool.afct.afct_pytorch import TorchAFct

from torch.utils.data.sampler import SubsetRandomSampler
from collections import deque

class DHLayer(torch.nn.Module):
    """
    This class represents a layer architecture based on DH Parameter as the learnable parameter
    and use them to construct the transformation matrix
    :param p_in: Number of Joints
    """

    def __init__(self, p_in):
        super(DHLayer, self).__init__()
        self._in = p_in

        self.register_parameter("alpha", torch.nn.Parameter((torch.rand(self._in, 1) - 0.5) * 1))
        self.register_parameter("beta", torch.nn.Parameter((torch.rand(self._in, 1) - 0.5) * 1))
        self.register_parameter("a", torch.nn.Parameter((torch.rand(self._in, 1) - 0.5) * 1))
        self.register_parameter("b", torch.nn.Parameter((torch.rand(self._in, 1) - 0.5) * 1))
        self.register_parameter("d", torch.nn.Parameter((torch.rand(self._in, 1) - 0.5) * 1))

    def forward(self, p_in):
        batch_size = p_in.shape[0]

        l_alpha = self.alpha.repeat(batch_size, 1).reshape(batch_size, self._in)
        l_beta = self.beta.repeat(batch_size, 1).reshape(batch_size, self._in)
        l_tx = self.a.repeat(batch_size, 1, 1).reshape(batch_size, self._in, 1, 1)
        l_ty = self.b.repeat(batch_size, 1, 1).reshape(batch_size, self._in, 1, 1)
        l_tz = self.d.repeat(batch_size, 1, 1).reshape(batch_size, self._in, 1, 1)

        # Construct Z Transformations
        unit = torch.Tensor([[0.0, 0.0, 1.0]], device=p_in.device)
        trans_z = self.construct_transformation(unit, p_in, l_tz)

        # Construct Y Transformations
        unit = torch.Tensor([[0.0, 1.0, 0.0]])
        trans_y = self.construct_transformation(unit, l_beta, l_ty)

        # Construct X Transformations
        unit = torch.Tensor([[1.0, 0.0, 0.0]], device=p_in.device)
        trans_x = self.construct_transformation(unit, l_alpha, l_tx)

        # Construct DH Matrix
        # dh_mat = trans_z * trans_y * trans_x
        dh_mat = torch.matmul(trans_z, trans_y)
        dh_mat = torch.matmul(dh_mat, trans_x)

        return dh_mat

    def construct_transformation(self, p_unit, p_angle, p_transl):
        """
        Construct Transformation Matrix
        :param p_unit: Rotation and Translation Unit Vector
        :param p_angle: Rotation Angle
        :param p_transl: Translation
        :return: trans_mat: Transformations Matrix
        """

        # Create Rotation Matrix
        rot_mat = self.create_rot_mat(p_unit.to(p_angle.device), p_angle)

        # Create Translation Vector
        transl_vec = p_unit * p_transl

        # Combine Rotation Matrix and Translation Vector into Transformation Matrix
        trans_mat = self.trans_rot_to_transformations_mat(transl_vec, rot_mat)

        return trans_mat

    def trans_rot_to_transformations_mat(self, p_trans, p_rot):
        """
        Combine Translation Vector and Rotation Matrix into Transformation Matrix
        :param p_trans: Translation Vector
        :param p_rot: Rotation Matrix
        :return: trans_mat: Transformation Matrix
        """
        batch_size = p_trans.shape[0]
        fixed_vec = torch.Tensor([[0.0, 0.0, 0.0, 1.0]], device=p_trans.device)
        trans_mat = torch.cat([p_rot, p_trans.permute(0, 1, 3, 2)], dim=3)
        trans_mat = torch.cat([trans_mat, fixed_vec.repeat(batch_size, self._in, 1, 1)], dim=2)
        return trans_mat

    def create_trans_vec(self):
        pass

    def create_rot_mat(self, p_unit, p_angle):
        """
        Create Rotation Matrix
        :param p_unit: Rotation Unit Vector
        :param p_angle: Rotation Angle
        :return: rot_mat: Rotation Matrix
        """
        batch_size = p_angle.shape[0]
        masking_indices = torch.tensor([[3, 2, 1], [2, 3, 0], [1, 0, 3]], device=p_angle.device)
        masking_cross = torch.Tensor([[0, -1, 1], [1, 0, -1], [-1, 1, 0]], device=p_angle.device)

        unit = p_unit
        unit_stack = torch.Tensor([], device=p_angle.device)
        outer_prod = torch.Tensor([], device=p_angle.device)

        for i in range(self._in):
            outer_prod = torch.cat([outer_prod, torch.ger(unit[0], unit[0])])
            k_unit = torch.cat([unit, torch.Tensor([[0.0]], device=p_angle.device)], dim=1)
            unit_store = k_unit[0][masking_indices] * masking_cross
            unit_stack = torch.cat([unit_stack, unit_store])

        unit = unit_stack.reshape(self._in, 3, 3).repeat(batch_size, 1, 1, 1)
        outer = outer_prod.reshape(self._in, 3, 3).repeat(batch_size, 1, 1, 1)

        rot_mat = unit * torch.sin(p_angle).reshape(batch_size, self._in, 1, 1) + (
                torch.eye(3, device=p_angle.device).repeat(batch_size, 1, 1, 1) - outer) * \
                  torch.cos(p_angle).reshape(batch_size, self._in, 1, 1) + outer

        return rot_mat

class N3(torch.nn.Module):
    def __init__(self, n_in):
        super(N3, self).__init__()
        self.n_in = n_in
        self.added = 0

    def forward(self, I):
        BatchSize = I.shape[0]

        A = torch.eye(4)
        b = torch.tril(torch.ones(self.n_in + self.added + 1, self.n_in + self.added)).flatten()
        c = torch.triu(torch.ones(self.n_in + self.added + 1, self.n_in + self.added), diagonal=1).flatten()

        maskIlower = torch.einsum('ij,k->kij', A, b).reshape(self.n_in + self.added + 1, self.n_in + self.added, 4, 4)
        maskIupper = torch.einsum('ij,k->kij', A, c).reshape(self.n_in + self.added + 1, self.n_in + self.added, 4, 4)

        output1 = torch.matmul(maskIlower, I.reshape(BatchSize, 1, self.n_in + self.added, 4, 4))
        output1 = torch.add(output1, maskIupper)
        output1 = output1.permute(0, 2, 1, 3, 4)

        output = torch.eye(4).repeat(BatchSize, self.n_in + self.added + 1, 1, 1)
        for outnum in range(self.n_in + self.added):
            output = torch.matmul(output, output1[:, outnum])

        return output

class RobotHTMModel(torch.nn.Module):
    """
    Provide Forward Kinematic based on Neural Network with DH Layer.
    Predicts the end-effector position with given joint angles.
    """

    def __init__(self, p_in, p_t):
        super(RobotHTMModel, self).__init__()
        self._in = p_in
        self._t = p_t

        self.dh_layer = DHLayer(self._in)
        self.eef_layer = N3(self._in)

    def forward(self, p_in):
        batch_size=p_in.shape[0]
        new_i = p_in.reshape(batch_size,2,self._in) * torch.cat([torch.Tensor([self._t]).repeat(1,self._in), torch.ones(1,self._in)])
        new_i = torch.sum(new_i,dim=1)
        
        trans = self.dh_layer(new_i)
        output = self.eef_layer(trans)
        
        return output

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
        self.net_model = RobotHTMModel(self.joint_num, 0.01)
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
            angle = torch.Tensor(transformations.euler_from_matrix(p_output[-1][idx][:].detach().numpy(), 'rxyz')) - thets
            thets = torch.Tensor(transformations.euler_from_matrix(p_output[-1][idx][:].detach().numpy(), 'rxyz'))
            angles = torch.cat([angles, torch.norm(angle).reshape(1, 1)], dim=1)

        output = torch.cat([self.input_temp, p_output[-1][-1][:3, [-1]].reshape(1,3)], dim=1)
        output = torch.cat([output, angles], dim=1)

        return output
    
    def _adapt(self, p_input: Element, p_output: Element) -> bool:
        model_input = deque(p_input.get_values()[6:])
        model_input.rotate(self.joint_num)
        model_input = torch.Tensor([list(model_input)])

        self.sim_env.set_theta(torch.Tensor([p_output.get_values()[6 : 6 + self.joint_num]]))
        self.sim_env.update_joint_coords()

        model_output = self.sim_env.get_homogeneous().reshape(self.joint_num+1,4,4)

        self._add_buffer(IOElement(model_input, model_output))

        if self._buffer.get_internal_counter() % 100 != 0:
            return False

        # Divide Test and Train
        if self.train_model:
            dataset_size = len(self._buffer)
            indices = list(range(dataset_size))
            split = int(np.floor(0.2 * dataset_size))
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

class HTMEnvModel(RobotHTM, EnvModel):
    C_NAME = "HTM Env Model"

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

