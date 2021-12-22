## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - A Synoptic Framework for Standardized Machine Learning Tasks
## -- Package : mlpro.rl.envmodels
## -- Module  : mlpenvmodel
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2021-12-17  0.0.0     MRD      Creation
## -- 2021-12-17  1.0.0     MRD      Released first version
## -- 2021-12-20  1.0.1     DA       Replaced 'done' by 'success'
## -- 2021-12-21  1.0.2     DA       Class MLPEnvMdel: renamed method reset() to _reset()
## -------------------------------------------------------------------------------------------------

"""
Ver. 1.0.2 (2021-12-21)

This module provides Environment Model based on MLP Neural Network for
robotinhtm environment.
"""

import torch

from mlpro.rl.models import *
from mlpro.rl.pool.envs.robotinhtm import RobotArm3D
from mlpro.rl.pool.envs.robotinhtm import RobotHTM
from mlpro.sl.pool.afct.mlpafctrans import MLPAFctTrans


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

        self.RobotArmSim = copy.deepcopy(self.RobotArm1)

        # Setup space
        # 1 Setup state space
        obs_space = ESpace()

        obs_space.add_dim(Dimension(0, "Tx", "Targetx", "", "m", "m", p_boundaries=[-np.inf, np.inf]))
        obs_space.add_dim(Dimension(1, "Ty", "Targety", "", "m", "m", p_boundaries=[-np.inf, np.inf]))
        obs_space.add_dim(Dimension(2, "Tz", "Targetz", "", "m", "m", p_boundaries=[-np.inf, np.inf]))
        obs_space.add_dim(Dimension(3, "Px", "Targetx", "", "m", "m", p_boundaries=[-np.inf, np.inf]))
        obs_space.add_dim(Dimension(4, "Py", "Targety", "", "m", "m", p_boundaries=[-np.inf, np.inf]))
        obs_space.add_dim(Dimension(5, "Pz", "Targetz", "", "m", "m", p_boundaries=[-np.inf, np.inf]))

        for idx in range(self.num_joint):
            obs_space.add_dim(
                Dimension(idx + 6, "J%i" % (idx), "Joint%i" % (idx), "", "deg", "deg", p_boundaries=[-np.inf, np.inf])
            )

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
            MLPAFctTrans,
            p_state_space=obs_space,
            p_action_space=action_space,
            p_threshold=-1,
            p_buffer_size=10000,
            p_ada=p_ada,
            p_logging=p_logging,
            p_sim_env=self.RobotArmSim,
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
        self._state.set_success(True)

