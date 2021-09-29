## -----------------------------------------------------------------------------
## -- Project : FH-SWF Automation Technology - Common Code Base (CCB)
## -- Package : mlpro
## -- Module  : robotinhtm
## -----------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.  Auth.  Description
## -- 2021-09-10  0.00  MRD    Creation
## -- 2021-09-11  1.00  MRD    Release of first version
## -- 2021-09-11  1.01  MRD    Change Header information to match our new library name
## -- 2021-09-25  1.02  MRD    Minor fix for state space and action space recognition
## -- 2021-09-30  1.03  MRD    Change from math.inf to np.inf
## -----------------------------------------------------------------------------

"""
Ver. 1.03 (2021-09-30)

This module provide an environment of a robot manipulator based on Homogeneous Matrix
"""
import torch
import math
import numpy as np
import random
from mlpro.rl.models import *
from mlpro.gt.models import *

## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class RobotArm3D:
    """
    Auxilary class for the implementation of robotinhtm.
    Generate the Kinematic of a pre-defined robot in Homogeneous Matrix.
    """
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

    def add_link_joint(self, jointAxis=None, lvector=None, thetaInit=None, adjustRot=None, adjustTheta=None):
        self.joints = torch.cat([self.joints, torch.Tensor([[0,0,0,1]]).T], dim=1)
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


    def get_transformation_matrix(self, theta, lvector, rotAxis, adjustRots=None, adjustThetas=None):
        transformationMatrix = torch.Tensor([])
            
        if rotAxis == 'rx':
            rotM = torch.Tensor([
                [1, 0, 0],
                [0, torch.cos(theta), -torch.sin(theta)],
                [0, torch.sin(theta), torch.cos(theta)]
                ])
        elif rotAxis == 'ry':
            rotM = torch.Tensor([
                [torch.cos(theta), 0, torch.sin(theta)],
                [0, 1, 0],
                [-torch.sin(theta), 0, torch.cos(theta)]
                ])
        elif rotAxis == 'rz':
            rotM = torch.Tensor([
                [torch.cos(theta), -torch.sin(theta), 0],
                [torch.sin(theta), torch.cos(theta), 0],
                [0, 0, 1]
                ])
        elif rotAxis == 'f':
            rotM = torch.eye(3)

        # Adjust Rotation Projection
        if adjustRots is None:
            rotM = torch.matmul(rotM,torch.eye(3))
        else:
            rotMsT = torch.eye(3)
            for projection in range(len(adjustRots)):
                if adjustRots[projection] == 'rx':
                    rotMs = torch.Tensor([
                        [1, 0, 0],
                        [0, torch.cos(adjustThetas[projection]), -torch.sin(adjustThetas[projection])],
                        [0, torch.sin(adjustThetas[projection]), torch.cos(adjustThetas[projection])]
                        ])
                elif adjustRots[projection] == 'ry':
                    rotMs = torch.Tensor([
                        [torch.cos(adjustThetas[projection]), 0, torch.sin(adjustThetas[projection])],
                        [0, 1, 0],
                        [-torch.sin(adjustThetas[projection]), 0, torch.cos(adjustThetas[projection])]
                        ])
                elif adjustRots[projection] == 'rz':
                    rotMs = torch.Tensor([
                        [torch.cos(adjustThetas[projection]), -torch.sin(adjustThetas[projection]), 0],
                        [torch.sin(adjustThetas[projection]), torch.cos(adjustThetas[projection]), 0],
                        [0, 0, 1]
                        ])
                rotMsT = torch.mm(rotMsT,rotMs)
            rotM = torch.mm(rotM,rotMsT)

        
        transformationMatrix = torch.cat([rotM, lvector.reshape(1,3).T], dim=1)
        transformationMatrix = torch.cat([transformationMatrix,torch.Tensor([[0,0,0,1]])], dim=0)
        return transformationMatrix

    def update_joint_coords(self):
        self.HM = torch.Tensor([])
        T = torch.eye(4)
        for i in range(len(self.lvector)):
            if self.jointAxis[i] != "f":
                T_next = self.get_transformation_matrix(
                self.thetas[i], self.lvector[i], rotAxis=self.jointAxis[i],
                adjustRots=self.adjustRot[i], adjustThetas=self.adjustTheta[i])
            else:
                T_next = self.get_transformation_matrix(
                torch.zeros(1), self.lvector[i], rotAxis=self.jointAxis[i],
                adjustRots=self.adjustRot[i], adjustThetas=self.adjustTheta[i])
            
            T = torch.mm(T,T_next)
            self.HM = torch.cat([self.HM, T])
            self.joints[:,[i]] = torch.mm(T,torch.Tensor([[0,0,0,1]]).T)

        # self.orientation = self.convert_to_quaternion_only()
        self.HMeef = T

    

    def get_joint(self):
        return self.joints

    def get_num_joint(self):
        return self.num_joint
    
    def set_theta(self, theta):
        self.thetas = theta.flatten()
        
    def get_homogeneous(self):
        return self.HM
    
    def get_homogeneous_eef(self):
        return self.HMeef

    def get_orientation(self):
        return self.orientation
    
    def update_theta(self, deltaTheta):
        self.thetas += deltaTheta.flatten()

## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class RobotHTM(Environment):
    """
    Custom environment for a arm robot represented as Homogeneous Matrix.
    The goal is to reach a certain point.
    """
    C_NAME      = 'RobotHTM'
    C_LATENCY   = timedelta(0,1,0)
    C_INFINITY  = np.finfo(np.float32).max

    def __init__(self, p_num_joints=3, p_target_mode="Random", p_logging=True):
        """
        Parameters:
            p_logging               Boolean switch for logging
            p_num_joints            Number of joints
            p_target_mode           Target Mode (Random or Fixed)
        """
        
        self.RobotArm1 = RobotArm3D()

        roboconf = {}
        roboconf["Joints"] = []

        jointType = []
        vectLinkLength = [[0,0,0],[0,0,0]]
        jointType.append("rz")
        for joint in range(p_num_joints-1):
            vectLinkLength.append([0,0.7,0])
            jointType.append("rx")

        jointType.append("f")

        for x in range(len(jointType)):
            vectorLink = dict(x=vectLinkLength[x][0],y=vectLinkLength[x][1],z=vectLinkLength[x][2])
            joint = dict(Joint_name="Joint %d" %x,Joint_type=jointType[x],Vector_link_length=vectorLink)
            roboconf["Joints"].append(joint)

        roboconf["Target_mode"] = p_target_mode
        roboconf["Update_rate"] = 0.01

        for robo in roboconf["Joints"]:
            self.RobotArm1.add_link_joint(lvector=torch.Tensor([[robo["Vector_link_length"]["x"],robo["Vector_link_length"]["y"],robo["Vector_link_length"]["z"]]]), jointAxis=robo["Joint_type"], thetaInit=torch.Tensor([math.radians(0)]))
        

        self.RobotArm1.update_joint_coords()
        self.jointangles = self.RobotArm1.thetas
        self.dt = roboconf["Update_rate"]
        self.mode = roboconf["Target_mode"]
        self.target = None
        self.init_distance = None
        self.num_joint = self.RobotArm1.get_num_joint()
        self.reach = torch.norm(torch.Tensor([[0.0, 0.0, 0.0]]) - self.RobotArm1.joints[:3,[-1]].reshape(1,3))
        self.last_distance = None

        super().__init__(p_mode=Environment.C_MODE_SIM,
                                p_logging=p_logging)

        if self.mode == "random":
            num = random.random()
            if num < 0.2:
                self.target = torch.Tensor([[0.5, 0.5, 0.5]])
                self.init_distance = torch.norm(torch.Tensor([[0.0, 0.0, 0.0]]) - self.target)
            elif num < 0.4:
                self.target = torch.Tensor([[0.0, 0.5, 0.5]])
                self.init_distance = torch.norm(torch.Tensor([[0.0, 0.0, 0.0]]) - self.target)
            elif num < 0.6:
                self.target = torch.Tensor([[-0.5, 0.0, 0.5]])
                self.init_distance = torch.norm(torch.Tensor([[0.0, 0.0, 0.0]]) - self.target)
            elif num < 0.8:
                self.target = torch.Tensor([[0.0, -0.5, 0.5]])
                self.init_distance = torch.norm(torch.Tensor([[0.0, 0.0, 0.0]]) - self.target)
            else:
                self.target = torch.Tensor([[-0.5, -0.5, 0.5]])
                self.init_distance = torch.norm(torch.Tensor([[0.0, 0.0, 0.0]]) - self.target)
        else:
            self.target = torch.Tensor([[0.5, 0.5, 0.5]])
            self.init_distance = torch.norm(torch.Tensor([[0.0, 0.0, 0.0]]) - self.target)
    
    def _setup_spaces(self):
        """
        Implement this method to enrich the state and action space with specific 
        dimensions. 
        """

        # 1 Setup state space
        self._state_space.add_dim(Dimension(0, 'Tx', 'Targetx', '', 'm', 'm', p_boundaries=[-np.inf,np.inf]))
        self._state_space.add_dim(Dimension(1, 'Ty', 'Targety', '', 'm', 'm', p_boundaries=[-np.inf,np.inf]))
        self._state_space.add_dim(Dimension(2, 'Tz', 'Targetz', '', 'm', 'm', p_boundaries=[-np.inf,np.inf]))
        self._state_space.add_dim(Dimension(3, 'Px', 'Positionx', '', 'm', 'm', p_boundaries=[-np.inf,np.inf]))
        self._state_space.add_dim(Dimension(4, 'Py', 'Positiony', '', 'm', 'm', p_boundaries=[-np.inf,np.inf]))
        self._state_space.add_dim(Dimension(5, 'Pz', 'Positionz', '', 'm', 'm', p_boundaries=[-np.inf,np.inf]))

        # 2 Setup action space
        for idx in range(self.num_joint):
            self._action_space.add_dim(Dimension(idx, 'J%i'%(idx), 'Joint%i'%(idx), '', 'rad/sec', '\frac{rad}{sec}', p_boundaries=[-np.pi,np.pi]))
    
    def _simulate_reaction(self, p_action: Action) -> None:
        action = p_action.get_sorted_values()
        if not isinstance(action, torch.Tensor):
            action = torch.Tensor(action)
        self.RobotArm1.update_theta(action*self.dt)
        self.RobotArm1.update_joint_coords()
        self.jointangles = self.RobotArm1.thetas
        self.state = self.get_state()
    
    def get_state(self) -> State:
        obs = torch.cat([self.target, self.RobotArm1.joints[:3,[-1]].reshape(1,3)], dim=1)
        obs = obs.cpu().numpy().flatten()
        state = State(self._state_space)
        state.set_values(obs)
        return state

   
    def _evaluate_state(self) -> None:
        disterror = np.linalg.norm(self.state.get_values()[:3] - self.state.get_values()[3:])
        if disterror <= 0.2:
            self.done = True
            self.goal_achievement = 1.0
        else:
            self.done = False
            self.goal_achievement = 0.0

    def compute_reward(self) -> Reward:
        reward = Reward(Reward.C_TYPE_OVERALL)
        disterror = np.linalg.norm(self.state.get_values()[:3] - self.state.get_values()[3:])
        rew = -disterror
        if disterror <= 0.2:
            rew = rew + 20
        reward.set_overall_reward(rew)
        return reward
    
    def set_theta(self,theta):
        self.RobotArm1.thetas = theta.reshape(self.num_joint)
        self.RobotArm1.update_joint_coords()
        self.jointangles = self.RobotArm1.thetas

    def reset(self):
        theta = torch.zeros(self.RobotArm1.get_num_joint())
        self.RobotArm1.set_theta(theta)
        self.RobotArm1.update_joint_coords()
        self.jointangles = self.RobotArm1.thetas
        if self.mode == "random":
            num = random.random()
            if num < 0.2:
                self.target = torch.Tensor([[0.5, 0.5, 0.5]])
                self.init_distance = torch.norm(torch.Tensor([[0.0, 0.0, 0.0]]) - self.target)
            elif num < 0.4:
                self.target = torch.Tensor([[0.0, 0.5, 0.5]])
                self.init_distance = torch.norm(torch.Tensor([[0.0, 0.0, 0.0]]) - self.target)
            elif num < 0.6:
                self.target = torch.Tensor([[-0.5, 0.0, 0.5]])
                self.init_distance = torch.norm(torch.Tensor([[0.0, 0.0, 0.0]]) - self.target)
            elif num < 0.8:
                self.target = torch.Tensor([[0.0, -0.5, 0.5]])
                self.init_distance = torch.norm(torch.Tensor([[0.0, 0.0, 0.0]]) - self.target)
            else:
                self.target = torch.Tensor([[-0.5, -0.5, 0.5]])
                self.init_distance = torch.norm(torch.Tensor([[0.0, 0.0, 0.0]]) - self.target)
        self.done = False
        self.goal_achievement = 0.0
        self.state = self.get_state()                
