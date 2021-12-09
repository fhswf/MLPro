import torch
import transformations

from mlpro.rl.models import *
from mlpro.rl.pool.envs.robotinhtm import RobotArm3D
from mlpro.rl.pool.afunctions.htmafctrans import HTMAFctTrans

# Create own MSpace
class MyTorchSpace(MSpace):
    def distance(self, p_e1: torch.Tensor, p_e2: torch.Tensor):
        output = torch.Tensor([])
        thets = torch.zeros(3)
        for joint in range(p_e1.shape[1] - 1):
            angle = torch.Tensor(transformations.euler_from_matrix(p_e1[-1][joint].detach().numpy(), "rxyz")) - thets
            thets = torch.Tensor(transformations.euler_from_matrix(p_e1[-1][joint].detach().numpy(), "rxyz"))
            output = torch.cat([output, torch.norm(angle).reshape(1, 1)], dim=1)

        return torch.norm(output - p_e2).item()


class HTMEnvModel(EnvModel):
    C_NAME = "HTM Env Model"

    def __init__(
        self,
        p_num_joints=4,
        p_target_mode="Random",
        p_ada=True,
        p_logging=True,
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

        self.RobotArmSim = copy.deepcopy(self.RobotArm1)

        # Setup space
        # 1 Setup state space
        obs_space = ESpace()
        for idx in range(self.num_joint):
            obs_space.add_dim(
                Dimension(idx, "J%i" % (idx), "Joint%i" % (idx), "", "deg", "deg", p_boundaries=[-np.inf, np.inf])
            )

        obs_space.add_dim(Dimension(idx + 1, "Tx", "Targetx", "", "m", "m", p_boundaries=[-np.inf, np.inf]))
        obs_space.add_dim(Dimension(idx + 2, "Ty", "Targety", "", "m", "m", p_boundaries=[-np.inf, np.inf]))
        obs_space.add_dim(Dimension(idx + 3, "Tz", "Targetz", "", "m", "m", p_boundaries=[-np.inf, np.inf]))

        # 2 Setup action space
        action_space = ESpace()
        for idx in range(self.num_joint):
            action_space.add_dim(
                Dimension(
                    idx,
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
            HTMAFctTrans,
            p_state_space=obs_space,
            p_action_space=action_space,
            p_input_space_cls=MyTorchSpace,
            p_output_space_cls=MyTorchSpace,
            p_threshold=0.01,
            p_buffer_size=1000,
            p_ada=p_ada,
            p_logging=p_logging,
        )

        super().__init__(
            p_observation_space=obs_space,
            p_action_space=action_space,
            p_latency=self.dt,
            p_afct_strans=afct_strans,
            p_afct_reward=None,
            p_afct_done=None,
            p_afct_broken=None,
            p_ada=p_ada,
            p_logging=p_logging,
        )

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

    def set_theta(self, theta):
        self.RobotArm1.thetas = theta.reshape(self.num_joint)
        self.RobotArm1.update_joint_coords()
        self.jointangles = self.RobotArm1.thetas

    def reset(self, p_seed=None) -> None:
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
        obs = torch.cat([self.RobotArm1.thetas.reshape(1, self.num_joint), self.target], dim=1)
        obs = obs.cpu().numpy().flatten()
        self._state = State(self._state_space)
        self._state.set_values(obs)
        self._state.set_done(True)
