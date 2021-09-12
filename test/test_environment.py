## -------------------------------------------------------------------------------------------------
## -- Project : FH-SWF Automation Technology - Common Code Base (CCB)
## -- Package : mlpro
## -- Module  : test_environment
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2021-09-11  1.0.0     MRD      Creation
## -- 2021-09-11  1.0.0     MRD      Release First Version
## -------------------------------------------------------------------------------------------------

"""
Ver. 1.0.0 (2021-09-11)

Unit test classes for environment.
"""


import unittest
import random
import numpy as np
from mlpro.rl.models import *
from mlpro.rl.pool.envs.robotinhtm import RobotHTM
from mlpro.rl.pool.envs.bglp import BGLP
from mlpro.rl.pool.envs.gridworld import GridWorld
## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class TestEnvironment(unittest.TestCase):
    """
    Unit tests for environment.
    """
## -------------------------------------------------------------------------------------------------
    def setUp(self):
        # Instatiate the environmnet
        self.env_cls = RobotHTM(3, p_logging=False)
        # self.env_cls = BGLP()
        # self.env_cls = GridWorld()

## -------------------------------------------------------------------------------------------------
    def test_type_class(self):
        self.assertIsInstance(self.env_cls, Environment)

## -------------------------------------------------------------------------------------------------
    def test_state_space(self):
        self.assertIsInstance(self.env_cls.get_state_space(), ESpace, "The type of state space is wrong")
        self.assertNotEqual(self.env_cls.get_state_space().get_num_dim(), 0, "The state space dimension is still empty")

    def test_action_space(self):
        self.assertIsInstance(self.env_cls.get_action_space(), ESpace, "The type of action space is wrong")
        self.assertNotEqual(self.env_cls.get_action_space().get_num_dim(), 0, "The action space dimenstion is still empty")

    def test_state(self):
        self.assertIsInstance(self.env_cls.get_state(), State, "The type of state is wrong")

    def test_simulate_reaction(self):
        my_action_values = np.zeros(self.env_cls.get_action_space().get_num_dim())
        for d in range(self.env_cls.get_action_space().get_num_dim()):
            my_action_values[d] = random.random() 
        
        my_action_values = Action(0, self.env_cls.get_action_space(), my_action_values)
        self.env_cls.process_action(my_action_values)

    def test_compute_reward(self):
        my_action_values = np.zeros(self.env_cls.get_action_space().get_num_dim())
        for d in range(self.env_cls.get_action_space().get_num_dim()):
            my_action_values[d] = random.random() 
        
        my_action_values = Action(0, self.env_cls.get_action_space(), my_action_values)
        self.env_cls.process_action(my_action_values)

        self.assertIsInstance(self.env_cls.compute_reward(), Reward, "The type of the reward is wrong")

    def test_complete_runtime(self):
        state = self.env_cls.get_state()
        
        my_action_values = np.zeros(self.env_cls.get_action_space().get_num_dim())
        for d in range(self.env_cls.get_action_space().get_num_dim()):
            my_action_values[d] = random.random() 
        
        my_action_values = Action(0, self.env_cls.get_action_space(), my_action_values)
        
        self.env_cls.process_action(my_action_values)
        
        reward = self.env_cls.compute_reward()

        self.env_cls._evaluate_state()

        self.env_cls.reset()

    


if __name__ == '__main__': unittest.main()
