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
## -------------------------------------------------------------------------------------------------

"""
Ver. 2.0.1 (2022-02-25)

This module provides Environment Model based on MLP Neural Network for
robotinhtm environment.
"""

import torch
import transformations

from mlpro.rl.models import *
from mlpro.rl.pool.envs.robotinhtm import RobotArm3D
from mlpro.rl.pool.envs.robotinhtm import RobotHTM
from mlpro.sl.pool.afct.afctrans_pytorch import TorchAFctTrans

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

class RobothtmAFct(TorchAFctTrans):
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

class MLPEnvModel(EnvModel, Mode):
    C_NAME = "HTM Env Model"

    def __init__(
        self,
        p_num_joints=4,
        p_target_mode="Random",
        p_ada=True,
        p_logging=False,
    ):

        # Define all the adaptive function here
        self.RobotArm1 = RobotArm3D()

        roboconf = {}
        roboconf["Joints"] = []

        jointType = []
        vectLinkLength = [[0, 0, 0], [0, 0, 0]]
        jointType.append("rz")
        for joint in range(p_num_joints - 1):
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
            roboconf["Joints"].append(joint)

        roboconf["Target_mode"] = p_target_mode
        roboconf["Update_rate"] = 0.01

        for robo in roboconf["Joints"]:
            self.RobotArm1.add_link_joint(
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

        self.RobotArm1.update_joint_coords()
        self.jointangles = self.RobotArm1.thetas
        self.dt = roboconf["Update_rate"]
        self.modes = roboconf["Target_mode"]
        self.target = None
        self.init_distance = None
        self.num_joint = self.RobotArm1.get_num_joint()
        self.reach = torch.norm(torch.Tensor([[0.0, 0.0, 0.0]]) - self.RobotArm1.joints[:3, [-1]].reshape(1, 3))

        # Setup space
        # 1 Setup state space
        obs_space = ESpace()

        obs_space.add_dim(Dimension("Tx", "Targetx", "", "m", "m", p_boundaries=[-np.inf, np.inf]))
        obs_space.add_dim(Dimension("Ty", "Targety", "", "m", "m", p_boundaries=[-np.inf, np.inf]))
        obs_space.add_dim(Dimension("Tz", "Targetz", "", "m", "m", p_boundaries=[-np.inf, np.inf]))
        obs_space.add_dim(Dimension("Px", "Targetx", "", "m", "m", p_boundaries=[-np.inf, np.inf]))
        obs_space.add_dim(Dimension("Py", "Targety", "", "m", "m", p_boundaries=[-np.inf, np.inf]))
        obs_space.add_dim(Dimension("Pz", "Targetz", "", "m", "m", p_boundaries=[-np.inf, np.inf]))

        for idx in range(self.num_joint):
            obs_space.add_dim(
                Dimension("J%i" % (idx), "Joint%i" % (idx), "", "deg", "deg", p_boundaries=[-np.inf, np.inf])
            )

        # 2 Setup action space
        action_space = ESpace()
        for idx in range(self.num_joint):
            action_space.add_dim(
                Dimension(
                    "A%i" % (idx),
                    "AV%i" % (idx),
                    "",
                    "rad/sec",
                    "\frac{rad}{sec}",
                    p_boundaries=[-np.pi, np.pi],
                )
            )

        # Setup Adaptive Function
        # HTM Function Here
        afct_strans = AFctSTrans(
            RobothtmAFct,
            p_state_space=obs_space,
            p_action_space=action_space,
            p_threshold=-1,
            p_buffer_size=10000,
            p_ada=p_ada,
            p_logging=p_logging,
        )

        EnvModel.__init__(
            self,
            p_observation_space=obs_space,
            p_action_space=action_space,
            p_latency=timedelta(seconds=self.dt),
            p_afct_strans=afct_strans,
            p_afct_reward=None,
            p_afct_success=None,
            p_afct_broken=None,
            p_ada=p_ada,
            p_logging=p_logging,
        )

        Mode.__init__(self, p_mode=Mode.C_MODE_SIM, p_logging=p_logging)

        if self.modes == "random":
            num = random.random()
            if num < 0.2:
                self.target = torch.Tensor([[0.5, 0.5, 0.5]])
                self.init_distance = torch.norm(self.RobotArm1.joints[:3, [-1]].reshape(1, 3) - self.target)
            elif num < 0.4:
                self.target = torch.Tensor([[0.0, 0.5, 0.5]])
                self.init_distance = torch.norm(self.RobotArm1.joints[:3, [-1]].reshape(1, 3) - self.target)
            elif num < 0.6:
                self.target = torch.Tensor([[-0.5, 0.0, 0.5]])
                self.init_distance = torch.norm(self.RobotArm1.joints[:3, [-1]].reshape(1, 3) - self.target)
            elif num < 0.8:
                self.target = torch.Tensor([[0.0, -0.5, 0.5]])
                self.init_distance = torch.norm(self.RobotArm1.joints[:3, [-1]].reshape(1, 3) - self.target)
            else:
                self.target = torch.Tensor([[-0.5, -0.5, 0.5]])
                self.init_distance = torch.norm(self.RobotArm1.joints[:3, [-1]].reshape(1, 3) - self.target)
        else:
            self.target = torch.Tensor([[0.5, 0.5, 0.5]])
            self.init_distance = torch.norm(self.RobotArm1.joints[:3, [-1]].reshape(1, 3) - self.target)

        self.reset()

    ## -------------------------------------------------------------------------------------------------
    def _compute_success(self, p_state: State = None) -> bool:
        # disterror = np.linalg.norm(p_state.get_values()[:3] - p_state.get_values()[3:6])
        disterror = np.linalg.norm(np.array(p_state.get_values())[:3] - np.array(p_state.get_values())[3:6])
        if disterror <= 0.1:
            self._state.set_terminal(True)
            return True
        else:
            return False

    ## -------------------------------------------------------------------------------------------------
    def _compute_broken(self, p_state: State) -> bool:
        return False

    ## -------------------------------------------------------------------------------------------------
    def _compute_reward(self, p_state_old: State, p_state_new: State) -> Reward:
        reward = Reward(self.C_REWARD_TYPE)
        # disterror = np.linalg.norm(p_state_new.get_values()[:3] - p_state_new.get_values()[3:6])
        disterror = np.linalg.norm(np.array(p_state_new.get_values())[:3] - np.array(p_state_new.get_values())[3:6])

        ratio = disterror / self.init_distance.item()
        rew = -np.ones(1) * ratio
        rew = rew - 10e-2
        if disterror <= 0.1:
            rew = rew + 1
        rew = rew.astype("float64")
        reward.set_overall_reward(rew)
        return reward

    def set_theta(self, theta):
        self.RobotArm1.thetas = theta.reshape(self.num_joint)
        self.RobotArm1.update_joint_coords()
        self.jointangles = self.RobotArm1.thetas

    def _reset(self, p_seed=None) -> None:
        self.set_random_seed(p_seed)
        theta = torch.zeros(self.RobotArm1.get_num_joint())
        self.RobotArm1.set_theta(theta)
        self.RobotArm1.update_joint_coords()
        self.jointangles = self.RobotArm1.thetas
        if self.modes == "random":
            num = random.random()
            if num < 0.2:
                self.target = torch.Tensor([[0.5, 0.5, 0.5]])
                self.init_distance = torch.norm(self.RobotArm1.joints[:3, [-1]].reshape(1, 3) - self.target)
            elif num < 0.4:
                self.target = torch.Tensor([[0.0, 0.5, 0.5]])
                self.init_distance = torch.norm(self.RobotArm1.joints[:3, [-1]].reshape(1, 3) - self.target)
            elif num < 0.6:
                self.target = torch.Tensor([[-0.5, 0.0, 0.5]])
                self.init_distance = torch.norm(self.RobotArm1.joints[:3, [-1]].reshape(1, 3) - self.target)
            elif num < 0.8:
                self.target = torch.Tensor([[0.0, -0.5, 0.5]])
                self.init_distance = torch.norm(self.RobotArm1.joints[:3, [-1]].reshape(1, 3) - self.target)
            else:
                self.target = torch.Tensor([[-0.5, -0.5, 0.5]])
                self.init_distance = torch.norm(self.RobotArm1.joints[:3, [-1]].reshape(1, 3) - self.target)
        obs = torch.cat(
            [
                self.target,
                self.RobotArm1.joints[:3, [-1]].reshape(1, 3),
                self.RobotArm1.thetas.reshape(1, self.num_joint),
            ],
            dim=1,
        )
        obs = obs.cpu().flatten().tolist()
        self._state = State(self._state_space)
        self._state.set_values(obs)

