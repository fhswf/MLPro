## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - The integrative middleware framework for standardized machine learning
## -- Package : mlpro.rl.pool.envs
## -- Module  : collisionavoidance_2D.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2024-07-10  0.0.0     MRD/SY   Creation
## -- 2024-07-10  1.0.0     MRD/SY   Release of first version
## -- 2024-07-12  1.0.1     SY       Add initial and target points into the state space
## -- 2024-07-16  1.0.2     SY       Update _compute_broken() method
## -- 2025-07-17  1.1.0     DA       Refactoring
## -------------------------------------------------------------------------------------------------

"""
Ver. 1.1.0 (2025-07-17) 

This module provides a 2D environment for collision avoidance of a trajectory planning with
dynamic goals. The DynamicTrajectoryPlanner environment simulates a 2D space where an agent must
navigate towards dynamically moving goals while avoiding static obstacles.

The agent, equipped with sensors providing information only on initial to goal trajectories,
operates within a continuous action space to move and adjust its path.

In this module, rewards are not defined but we give some important components for developing the
reward function including actual distance of the trajectory and number of collisions that can be used
to penalise for any collisions.

This environment is dynamic, with goals potentially changing over time, which makes it ideal for
training reinforcement learning agents for tasks like autonomous robot navigation, drone flight in urban
areas, and autonomous vehicle path planning, which emphasize trajectory optimization and collision avoidance
in real-world-like settings.

"""

import random
import numpy as np

from matplotlib import pyplot as plt
from matplotlib.patches import Polygon
from matplotlib import path

from mlpro.bf import Log, ParamError
from mlpro.bf.math import Dimension, ESpace
from mlpro.bf.systems import State, Action

from mlpro.rl.models import *
      


# Export list for public API
__all__ = [ 'DynamicTrajectoryPlanner' ]



        
## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class DynamicTrajectoryPlanner(Environment):
    """
    This is the main class of Dynamic Trajectory Planner environment that inherits Environment
    class from MLPro.
    
    Parameters
    ----------
    p_visualize : bool, optional
        Boolean switch for visualisation. Default = False.
    p_logging : Log, optional
        Logging functionalities. Default = Log.C_LOG_ALL.
    p_num_point : int, optional
        Number of nodes within the trajectory, including start and finish nodes. Default = 5.
    p_cycle_limit : int, optional
        Cycle limit before the environment ends. Default = 100.
    p_xlimit : list, optional
        Limit of x-axis of the environment frame. Default = [-4,4].
    p_ylimit : list, optional
        Limit of y-axis of the environment frame. Default = [-4,4].
    p_action_boundaries : list, optional
        Action boundaries. Default = [-0.02,0.02].
    p_dt : float, optional
        Plot pause duration. Default = 0.01.
    p_resolution : float, optional
        Plot resolution. Default = 0.01.
    p_multi_goals : list, optional
        Set a list of possible goals (target nodes) positions. Default = [[3,3],[2,2],[1,3],[3,1]].
    p_start_pos : list, optional
        Starting node position. Default = [-3,-3].
    p_obstacles : list, optional
        Obstacles positions. Default = [[-1,-1],[1,-1],[1,1],[-1,1]].
        
    """
    
    C_NAME          = 'Trajectory Planner'
    C_CYCLE_LIMIT   = 0
    
    
## -------------------------------------------------------------------------------------------------
    def __init__( self,
                  p_visualize:bool = True,
                  p_logging = Log.C_LOG_ALL,
                  p_num_point:int = 5,
                  p_cycle_limit:int = 100,
                  p_xlimit:list = [-4,4],
                  p_ylimit:list = [-4,4],
                  p_action_boundaries:list = [-0.02,0.02],
                  p_dt:float = 0.01,
                  p_resolution:float = 0.01,
                  p_multi_goals:list = [[3,3],[2,2],[1,3],[3,1]],
                  p_start_pos:list = [-3,-3],
                  p_obstacles:list = [[-1,-1],[1,-1],[1,1],[-1,1]]):

        if p_num_point <= 2:
            raise ParamError("p_num_point must be bigger than 2.")
        else:
            self.num_traject_point  = p_num_point
            
        self.resolution         = p_resolution
        self.C_CYCLE_LIMIT      = p_cycle_limit
        self.start_pos          = np.array(p_start_pos)
        self.multi_goals        = p_multi_goals
        self.x_limit            = p_xlimit
        self.y_limit            = p_ylimit
        self.action_boundaries  = p_action_boundaries
        self._plot_avail        = False

        self.obstacles = [
            np.array(p_obstacles),
            np.array([[self.x_limit[0], self.y_limit[0]],[self.x_limit[0]+0.1, self.y_limit[0]],[self.x_limit[0]+0.1, self.y_limit[1]],[self.x_limit[0], self.y_limit[1]]]),
            np.array([[self.x_limit[1]-0.1, self.y_limit[0]],[self.x_limit[1], self.y_limit[0]],[self.x_limit[1], self.y_limit[1]],[self.x_limit[1]-0.1, self.y_limit[1]]]),
            np.array([[self.x_limit[0], self.y_limit[0]],[self.x_limit[1], self.y_limit[0]],[self.x_limit[1], self.y_limit[0]+0.1],[self.x_limit[0], self.y_limit[0]+0.1]]),
            np.array([[self.x_limit[0], self.y_limit[1]-0.1],[self.x_limit[1], self.y_limit[1]-0.1],[self.x_limit[1], self.y_limit[1]],[self.x_limit[0], self.y_limit[1]]]),
        ]

        super().__init__(p_mode=Environment.C_MODE_SIM, p_visualize=p_visualize, p_logging=p_logging)
        
        if self._visualize:
            self.dt = p_dt
            
        self.goal_pos = np.array(random.sample(self.multi_goals,1)[0])
        self.traject = None
        self.current_traject_plot = None
        self.collide_point_plot = None
        self.collide_line_plot = []
        self.start_plot = None
        self.goal_plot = None
        self.collide_line_list = []
        self.collide_point_list = []
        
        self._state_space, self._action_space = self._setup_spaces()
        self.reset()


## -------------------------------------------------------------------------------------------------
    @staticmethod
    def setup_spaces():
        
        return None, None

        
## -------------------------------------------------------------------------------------------------
    def _setup_spaces(self):

        state_space     = ESpace()
        action_space    = ESpace()

        for x in range(self.num_traject_point):
            state_space.add_dim(
                Dimension(
                    p_name_short='x_'+str(x),
                    p_base_set=Dimension.C_BASE_SET_R,
                    p_name_long='Node '+str(x)+' - x-axis',
                    p_boundaries=self.x_limit
                    )
                )
            state_space.add_dim(
                Dimension(
                    p_name_short='y_'+str(x),
                    p_base_set=Dimension.C_BASE_SET_R,
                    p_name_long='Node '+str(x)+' - y-axis',
                    p_boundaries=self.y_limit
                    )
                )

        for x in range(self.num_traject_point-2):
            action_space.add_dim(
                Dimension(
                    p_name_short='x_'+str(x+1),
                    p_base_set=Dimension.C_BASE_SET_R,
                    p_name_long='Node '+str(x+1)+' - x-axis',
                    p_boundaries=self.action_boundaries
                    )
                )
            action_space.add_dim(
                Dimension(
                    p_name_short='y_'+str(x+1),
                    p_base_set=Dimension.C_BASE_SET_R,
                    p_name_long='Node '+str(x+1)+' - y-axis',
                    p_boundaries=self.action_boundaries
                    )
                )

        return state_space, action_space
            
    
## -------------------------------------------------------------------------------------------------
    def _add_obstacle(self, p_obstacle):
        
        self.obstacles.append(p_obstacle)
            
    
## -------------------------------------------------------------------------------------------------
    def _draw_obstacle(self):
        
        for obstacle in self.obstacles:
            self.ax.add_patch(Polygon(obstacle))
                
        
## -------------------------------------------------------------------------------------------------
    def _draw_start_goal(self):
        
        self.ax.plot(self.start_pos[0], self.start_pos[1], 'o', color='blue', markersize=20)
        self.ax.plot(self.goal_pos[0], self.goal_pos[1], 'o', color='green', markersize=20)
            
    
## -------------------------------------------------------------------------------------------------
    def _draw_traj(self):
        
        self.start_plot = self.ax.plot([self.start_pos[0], self.traject[0,0]], [self.start_pos[1], self.traject[0,1]], '--', color='black')
        self.current_traject_plot = self.ax.plot(self.traject[:,0], self.traject[:,1], 'x--', color='black')
        self.goal_plot = self.ax.plot([self.goal_pos[0], self.traject[-1,0]], [self.goal_pos[1], self.traject[-1,1]], '--', color='black')
        if len(self.collide_point_list) > 0:
            self.collide_point_plot = self.ax.plot(self.collide_point_list[:,0], self.collide_point_list[:,1], 'x', color='red')
        if len(self.collide_line_list) > 0:
            for line in self.collide_line_list:
                self.collide_line_plot.append(self.ax.plot([line[0,0], line[1,0]], [line[0,1], line[1,1]], '--', color='red')[0])
                    
            
## -------------------------------------------------------------------------------------------------
    def _clear_traj(self):
        
        if self.start_plot and self.current_traject_plot and self.goal_plot:
            self.start_plot.pop(0).remove()
            self.current_traject_plot.pop(0).remove()
            self.goal_plot.pop(0).remove()
            if self.collide_point_plot:
                self.collide_point_plot.pop(0).remove()
            if self.collide_line_plot:
                for line in self.collide_line_plot:
                    line.remove()
                self.collide_line_plot = []
                    
            
## -------------------------------------------------------------------------------------------------
    def _calc_init_traj(self, p_num_point):
        
        xvals = np.linspace(self.start_pos[0], self.goal_pos[0], p_num_point)[1:-1]
        yvals = np.linspace(self.start_pos[1], self.goal_pos[1], p_num_point)[1:-1]
        self.traject = np.concatenate([xvals.reshape(1,p_num_point-2).T, yvals.reshape(1,p_num_point-2).T], axis=1)
            
    
## -------------------------------------------------------------------------------------------------
    def _collide_check_point_list(self):
        
        collide = False
        self.collide_point_list = []
        for point in self.traject:
            for obstacle in self.obstacles:
                hull = path.Path(obstacle)
                collide = hull.contains_points([point])
                if collide:
                    self.collide_point_list.append(point.tolist())
                    break

        self.collide_point_list = np.array(self.collide_point_list)
        return collide
            
    
## -------------------------------------------------------------------------------------------------
    def _collide_check_point(self, p_traject):
        
        collide = False
        for point in p_traject:
            for obstacle in self.obstacles:
                hull = path.Path(obstacle)
                collide = hull.contains_points([point])
                if collide:
                    return collide

        return collide
            
    
## -------------------------------------------------------------------------------------------------
    def _collide_check_line(self):
        
        collide = False
        self.collide_line_list = []

        trajectory = np.concatenate([
            [self.start_pos],
            self.traject,
            [self.goal_pos]
        ])

        for idx in range(len(trajectory)-1):
            collide_line = False
            if self.collide_point_list.tolist():
                if trajectory[idx] in self.collide_point_list or trajectory[idx+1] in self.collide_point_list:
                    self.collide_line_list.append([trajectory[idx].tolist(), trajectory[idx+1].tolist()])
                    continue
            if self.collide_line_list:
                if trajectory[idx] in np.array(self.collide_line_list) or trajectory[idx+1] in np.array(self.collide_line_list):
                    continue    
            l = np.linalg.norm(trajectory[idx+1] - trajectory[idx])
            num_point = int(l / self.resolution)
            x = np.linspace(trajectory[idx][0], trajectory[idx+1][0], num_point)[1:-1]
            y = np.linspace(trajectory[idx][1], trajectory[idx+1][1], num_point)[1:-1]
            line = np.concatenate([x.reshape(1,num_point-2).T, y.reshape(1,num_point-2).T], axis=1)
            collide_line = self._collide_check_point(line)
            if collide_line:
                self.collide_line_list.append([trajectory[idx].tolist(), trajectory[idx+1].tolist()])
        if self.collide_line_list:
            collide = True
        self.collide_line_list = np.array(self.collide_line_list)
        
        return collide   
        

## -------------------------------------------------------------------------------------------------
    def _get_obs(self):
        
        trajectory = np.concatenate([
            [self.start_pos],
            self.traject,
            [self.goal_pos]
        ])
        
        return trajectory.flatten()
        

## -------------------------------------------------------------------------------------------------
    def _calc_distance(self):
        
        trajectory = np.concatenate([
            [self.start_pos],
            self.traject,
            [self.goal_pos]
        ])
        
        distance = 0
        for idx in range(len(trajectory)-1):
            distance += np.linalg.norm(trajectory[idx] - trajectory[idx+1])

        return distance   
        

## -------------------------------------------------------------------------------------------------
    def _compute_reward(self, p_state_old:State, p_state_new:State) -> Reward:
        
        number_of_collide_points = 0
        number_of_collide_lines = 0
        for _ in self.collide_point_list:
            number_of_collide_points += 1
        for _ in self.collide_line_list:
            number_of_collide_lines += 1

        distance = self._calc_distance()
            
        reward = Reward()
        reward.set_overall_reward(0)
        
        return reward


## -------------------------------------------------------------------------------------------------
    def _compute_success(self, p_state:State) -> bool:
        
        collide_point = self._collide_check_point_list()
        collide_line = self._collide_check_line()
        
        if not collide_point and not collide_line:
            self._state.set_success(True)
            self._state.set_terminal(True)
            return True
        else:
            self._state.set_success(False)
            return False


## -------------------------------------------------------------------------------------------------
    def _compute_broken(self, p_state:State) -> bool:
        
        for pt in self.traject:
            if (pt[0]<self.x_limit[0]) or (pt[0]>self.x_limit[1]):
                self._state.set_broken(True)
                return True
            elif (pt[1]<self.y_limit[0]) or (pt[1]>self.y_limit[1]):
                self._state.set_broken(True)
                return True              
        
        self._state.set_broken(False)
        return False
            
    
## -------------------------------------------------------------------------------------------------
    def _reset(self, p_seed=None) -> None:
        
        random.seed(p_seed)
        np.random.seed(p_seed)
        
        self.goal_pos = np.array(random.sample(self.multi_goals,1)[0])
        
        if self._plot_avail:
            self.ax.clear()
            self._draw_start_goal()
            self._draw_obstacle()
            
        self._calc_init_traj(self.num_traject_point)
    
        if self._plot_avail:
            self.update_plot()
        
        obs = self._get_obs()
        self._state = State(self._state_space)
        for i in range(len(obs)):
            self._state.set_value(self._state.get_dim_ids()[i], obs[i])
        
        return self._get_obs()
        

## -------------------------------------------------------------------------------------------------
    def init_plot(self, p_figure=None):
        
        if not self._plot_avail:
            plt.ion()
            self.fig = plt.figure(figsize=(10, 10))
            self.ax = plt.gca()
            self.ax.set_xlim(self.x_limit)
            self.ax.set_ylim(self.y_limit)
            
            self.ax.clear()
            self._draw_start_goal()
            self._draw_obstacle()
            self.update_plot()
            
            self._plot_avail = True


## -------------------------------------------------------------------------------------------------
    def update_plot(self):
        
        self._clear_traj()
        self._draw_traj()
        plt.pause(self.dt)
        

## -------------------------------------------------------------------------------------------------
    def _simulate_reaction(self, p_state:State, p_action:Action) -> State:
        
        action = p_action.get_sorted_values()
        self.traject = self.traject+action.reshape((self.num_traject_point-2),2)
        
        if self._visualize:
            self.update_plot()
        
        obs = self._get_obs()
        self._state = State(self._state_space)
        for i in range(len(obs)):
            self._state.set_value(self._state.get_dim_ids()[i], obs[i])
        
        return self._state