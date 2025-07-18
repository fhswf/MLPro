## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - The integrative middleware framework for standardized machine learning
## -- Package : mlpro
## -- Module  : robotinhtm
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2021-09-10  0.0.0     MRD      Creation
## -- 2021-09-11  1.0.0     MRD      Release of first version
## -- 2021-09-11  1.0.1     MRD      Change Header information to match our new library name
## -- 2021-09-25  1.0.2     MRD      Minor fix for state space and action space recognition
## -- 2021-10-05  1.0.3     SY       Update following new attributes done and broken in State
## -- 2021-11-15  1.1.0     DA       Refactoring of class RobotHTM
## -- 2021-12-03  1.1.1     DA       Refactoring
## -- 2021-12-08  1.1.2     MRD      Change the state, include the joint angles
## -- 2021-12-19  1.1.3     DA       Replaced 'done' by 'success'
## -- 2021-12-21  1.1.4     DA       Class RobotHTM: renamed method reset() to _reset()
## -- 2021-12-21  1.1.5     MRD      Add Termination on Success
## -- 2022-01-21  1.1.6     MRD      Add recommended cycle limit and add seed parameter
## -- 2022-02-25  1.1.7     SY       Refactoring due to auto generated ID in class Dimension
## -- 2022-11-09  1.1.8     DA       Refactoring due to changes on plot systematics
## -- 2023-08-21  1.1.9     MRD      Remove Transformation package, and quaternion converter
## -- 2025-07-17  1.2.0     DA       Refactoring
## -------------------------------------------------------------------------------------------------

"""
Ver. 1.2.0 (2025-07-17) 

This module provides an environment of a robot manipulator based on Homogeneous Matrix
"""

from datetime import timedelta
import random

import numpy as np
import torch

from mlpro.bf import Log
from mlpro.bf.math import Dimension, ESpace     
from mlpro.bf.systems import State, Action 
from mlpro.rl.models import Reward, Environment



# Export list for public API
__all__ = [ 'RobotArm3D',
            'RobotHTM' ]




## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class RobotArm3D:
    """
    Auxiliary class for the implementation of robotinhtm.
    Generate the Kinematic of a pre-defined robot in Homogeneous Matrix.
    """

## -------------------------------------------------------------------------------------------------
    def __init__(self):
        self.thetas = torch.Tensor([])
        self.joints = torch.Tensor([])
        self.orientation = torch.Tensor([])
        self.adjustRot = []
        self.adjustTheta = []
        self.num_joint = 0
        self.jointAxis = []
        self.lvector = torch.Tensor([])
        self.HM = torch.Tensor([])
        self.HMeef = None


## -------------------------------------------------------------------------------------------------
    def add_link_joint(
            self,
            jointAxis=None,
            lvector=None,
            thetaInit=None,
            adjustRot=None,
            adjustTheta=None,
    ):
        self.joints = torch.cat([self.joints, torch.Tensor([[0, 0, 0, 1]]).T], dim=1)
        self.jointAxis.append(jointAxis)
        self.lvector = torch.cat([self.lvector, lvector])
        if adjustRot is not None:
            self.adjustRot.append(adjustRot)
            self.adjustTheta.append(adjustTheta)
        else:
            self.adjustRot.append(None)
            self.adjustTheta.append(None)
        if jointAxis != "f":
            self.thetas = torch.cat([self.thetas, thetaInit])
            self.num_joint = self.num_joint + 1


## -------------------------------------------------------------------------------------------------
    def get_transformation_matrix(self, theta, lvector, rotAxis, adjustRots=None, adjustThetas=None):
        transformationMatrix = torch.Tensor([])

        if rotAxis == "rx":
            rotM = torch.Tensor(
                [
                    [1, 0, 0],
                    [0, torch.cos(theta), -torch.sin(theta)],
                    [0, torch.sin(theta), torch.cos(theta)],
                ]
            )
        elif rotAxis == "ry":
            rotM = torch.Tensor(
                [
                    [torch.cos(theta), 0, torch.sin(theta)],
                    [0, 1, 0],
                    [-torch.sin(theta), 0, torch.cos(theta)],
                ]
            )
        elif rotAxis == "rz":
            rotM = torch.Tensor(
                [
                    [torch.cos(theta), -torch.sin(theta), 0],
                    [torch.sin(theta), torch.cos(theta), 0],
                    [0, 0, 1],
                ]
            )
        elif rotAxis == "f":
            rotM = torch.eye(3)

        # Adjust Rotation Projection
        if adjustRots is None:
            rotM = torch.matmul(rotM, torch.eye(3))
        else:
            rotMsT = torch.eye(3)
            for projection in range(len(adjustRots)):
                if adjustRots[projection] == "rx":
                    rotMs = torch.Tensor(
                        [
                            [1, 0, 0],
                            [
                                0,
                                torch.cos(adjustThetas[projection]),
                                -torch.sin(adjustThetas[projection]),
                            ],
                            [
                                0,
                                torch.sin(adjustThetas[projection]),
                                torch.cos(adjustThetas[projection]),
                            ],
                        ]
                    )
                elif adjustRots[projection] == "ry":
                    rotMs = torch.Tensor(
                        [
                            [
                                torch.cos(adjustThetas[projection]),
                                0,
                                torch.sin(adjustThetas[projection]),
                            ],
                            [0, 1, 0],
                            [
                                -torch.sin(adjustThetas[projection]),
                                0,
                                torch.cos(adjustThetas[projection]),
                            ],
                        ]
                    )
                elif adjustRots[projection] == "rz":
                    rotMs = torch.Tensor(
                        [
                            [
                                torch.cos(adjustThetas[projection]),
                                -torch.sin(adjustThetas[projection]),
                                0,
                            ],
                            [
                                torch.sin(adjustThetas[projection]),
                                torch.cos(adjustThetas[projection]),
                                0,
                            ],
                            [0, 0, 1],
                        ]
                    )
                rotMsT = torch.mm(rotMsT, rotMs)
            rotM = torch.mm(rotM, rotMsT)

        transformationMatrix = torch.cat([rotM, lvector.reshape(1, 3).T], dim=1)
        transformationMatrix = torch.cat([transformationMatrix, torch.Tensor([[0, 0, 0, 1]])], dim=0)
        return transformationMatrix


## -------------------------------------------------------------------------------------------------
    def update_joint_coords(self):
        self.HM = torch.Tensor([])
        T = torch.eye(4)
        for i in range(len(self.lvector)):
            if self.jointAxis[i] != "f":
                T_next = self.get_transformation_matrix(
                    self.thetas[i],
                    self.lvector[i],
                    rotAxis=self.jointAxis[i],
                    adjustRots=self.adjustRot[i],
                    adjustThetas=self.adjustTheta[i],
                )
            else:
                T_next = self.get_transformation_matrix(
                    torch.zeros(1),
                    self.lvector[i],
                    rotAxis=self.jointAxis[i],
                    adjustRots=self.adjustRot[i],
                    adjustThetas=self.adjustTheta[i],
                )

            T = torch.mm(T, T_next)
            self.HM = torch.cat([self.HM, T])
            self.joints[:, [i]] = torch.mm(T, torch.Tensor([[0, 0, 0, 1]]).T)

        # self.orientation = self.convert_to_quaternion_only()
        self.HMeef = T


## -------------------------------------------------------------------------------------------------
    def get_joint(self):
        return self.joints


## -------------------------------------------------------------------------------------------------
    def get_num_joint(self):
        return self.num_joint


## -------------------------------------------------------------------------------------------------
    def set_theta(self, theta):
        self.thetas = theta.flatten()


## -------------------------------------------------------------------------------------------------
    def get_homogeneous(self):
        return self.HM


## -------------------------------------------------------------------------------------------------
    def get_homogeneous_eef(self):
        return self.HMeef


## -------------------------------------------------------------------------------------------------
    def get_orientation(self):
        return self.orientation


## -------------------------------------------------------------------------------------------------
    def update_theta(self, deltaTheta):
        self.thetas += deltaTheta.flatten()





## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class RobotHTM (Environment):
    """
    Custom environment for an arm robot represented as Homogeneous Matrix.
    The goal is to reach a certain point.

    Parameters
    ----------
    p_num_joints : int
        Number of joints. Default = 4.
    p_reset_seed : bool
        If True, random generator is reset. Default = True.
    p_seed
        Seeding value for the random generator. Default = None.
    p_target_mode : str
        Target mode. Possible values are "random" (default) or "fixed".
    p_visualize : bool
        Boolean switch for env/agent visualisation. Default = True.
    p_logging
        Log level (see constants of class Log). Default = Log.C_LOG_ALL.
    """

    C_NAME          = "RobotHTM"
    C_REWARD_TYPE   = Reward.C_TYPE_OVERALL
    C_LATENCY       = timedelta(0, 1, 0)
    C_INFINITY      = np.finfo(np.float32).max
    C_CYCLE_LIMIT   = 100

## -------------------------------------------------------------------------------------------------
    def __init__( self, 
                  p_num_joints = 4, 
                  p_reset_seed = True, 
                  p_seed = None, 
                  p_target_mode : str = "random", 
                  p_visualize : bool = True,
                  p_logging = Log.C_LOG_ALL ):

        self.RobotArm1 = RobotArm3D()

        joints = []

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
            joints.append(joint)

        for robo in joints:
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
        self.dt = 0.01
        self.target_mode = p_target_mode
        self.target = None
        self.init_distance = None
        self.num_joint = self.RobotArm1.get_num_joint()
        self.reach = torch.norm(torch.Tensor([[0.0, 0.0, 0.0]]) - self.RobotArm1.joints[:3, [-1]].reshape(1, 3))
        self.last_distance = None

        super().__init__(p_mode=Environment.C_MODE_SIM, p_visualize=p_visualize, p_logging=p_logging)
        self._state_space, self._action_space = self._setup_spaces()
        self.set_random_seed(p_seed)
        self._reset_seed = p_reset_seed

        self.reset()


## -------------------------------------------------------------------------------------------------
    @staticmethod
    def setup_spaces():
        return None, None


## -------------------------------------------------------------------------------------------------
    def _setup_spaces(self):
        """
        Implement this method to enrich the state and action space with specific
        dimensions.
        """

        # 1 Setup state space
        state_space = ESpace()
        state_space.add_dim(Dimension("Tx", "Targetx", "", "m", "m", p_boundaries=[-np.inf, np.inf]))
        state_space.add_dim(Dimension("Ty", "Targety", "", "m", "m", p_boundaries=[-np.inf, np.inf]))
        state_space.add_dim(Dimension("Tz", "Targetz", "", "m", "m", p_boundaries=[-np.inf, np.inf]))
        state_space.add_dim(Dimension("Px", "Targetx", "", "m", "m", p_boundaries=[-np.inf, np.inf]))
        state_space.add_dim(Dimension("Py", "Targety", "", "m", "m", p_boundaries=[-np.inf, np.inf]))
        state_space.add_dim(Dimension("Pz", "Targetz", "", "m", "m", p_boundaries=[-np.inf, np.inf]))

        for idx in range(self.num_joint):
            state_space.add_dim(
                Dimension(
                    "J%i" % (idx),
                    "Joint%i" % (idx),
                    "",
                    "deg",
                    "deg",
                    p_boundaries=[-np.inf, np.inf],
                )
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

        return state_space, action_space


## -------------------------------------------------------------------------------------------------
    def _simulate_reaction(self, p_state: State, p_action: Action) -> State:
        action = p_action.get_sorted_values()
        if not isinstance(action, torch.Tensor):
            action = torch.Tensor(action)
        self.RobotArm1.update_theta(action * self.dt)
        self.RobotArm1.update_joint_coords()
        self.jointangles = self.RobotArm1.thetas

        obs = torch.cat(
            [
                self.target,
                self.RobotArm1.joints[:3, [-1]].reshape(1, 3),
                self.RobotArm1.thetas.reshape(1, self.num_joint),
            ],
            dim=1,
        )
        obs = obs.cpu().flatten().tolist()
        state = State(self._state_space)
        state.set_values(obs)

        return state


## -------------------------------------------------------------------------------------------------
    def _compute_success(self, p_state: State = None) -> bool:
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
        disterror = np.linalg.norm(np.array(p_state_new.get_values())[:3] - np.array(p_state_new.get_values())[3:6])

        ratio = disterror / self.init_distance.item()
        rew = -np.ones(1) * ratio
        rew = rew - 10e-2
        if disterror <= 0.1:
            rew = rew + 1
        rew = rew.astype("float64")
        reward.set_overall_reward(rew.item())
        return reward


## -------------------------------------------------------------------------------------------------
    def set_theta(self, theta):
        self.RobotArm1.thetas = theta.reshape(self.num_joint)
        self.RobotArm1.update_joint_coords()
        self.jointangles = self.RobotArm1.thetas


## -------------------------------------------------------------------------------------------------
    def _reset(self, p_seed=None) -> None:
        if self._reset_seed:
            self.set_random_seed(p_seed)
        theta = torch.zeros(self.RobotArm1.get_num_joint())
        self.RobotArm1.set_theta(theta)
        self.RobotArm1.update_joint_coords()
        self.jointangles = self.RobotArm1.thetas
        if self.target_mode == "random":
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