## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - A Synoptic Framework for Standardized Machine Learning Tasks
## -- Package : mlpro
## -- Module  : howto_gt_002_train_own_multi_player_with_multicartpole_game_board.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2021-06-06  0.0.0     DA       Creation
## -- 2021-06-06  1.0.0     DA       Release of first version
## -- 2021-07-01  1.1.0     DA       Extended by data logging/storing (user home directory)
## -- 2021-07-06  1.1.1     SY       Bugfix due to method Training.save_data() update
## -- 2021-08-28  1.1.2     DA       Adjustments after changings on rl models
## -- 2021-09-11  1.1.2     MRD      Change Header information to match our new library name
## -- 2021-09-28  1.1.3     SY       Adjustment due to implementation of SAR Buffer on player
## -- 2021-10-06  1.1.4     DA       Refactoring 
## -- 2021-11-16  1.2.0     DA       Refactoring 
## -- 2021-12-07  1.2.1     DA       Refactoring 
## -- 2022-02-25  1.2.2     SY       Refactoring due to auto generated ID in class Dimension
## -------------------------------------------------------------------------------------------------

"""
Ver. 1.2.2 (2021-02-25)
 
This module shows how to train an own multi-player with the enhanced multi-action
game board MultiCartPole based on the OpenAI Gym CartPole environment.
"""


from mlpro.rl.models import *
from mlpro.gt.models import *
from mlpro.gt.pool.boards.multicartpole import MultiCartPolePGT
import random
import numpy as np
from pathlib import Path
import os
from datetime import datetime





# 1 Implement your own agent policy
class MyPolicy(Policy):

    C_NAME      = 'MyPolicy'

    def compute_action(self, p_state: State) -> Action:
        # 1 Create a numpy array for your action values 
        my_action_values = np.zeros(self._action_space.get_num_dim())

        # 2 Computing action values is up to you...
        for d in range(self._action_space.get_num_dim()):
            my_action_values[d] = random.random() 

        # 3 Return an action object with your values
        return Action(self._id, self._action_space, my_action_values)


    def _adapt(self, *p_args) -> bool:
        # 1 Adapting the internal policy is up to you...
        self.log(self.C_LOG_TYPE_I, 'Sorry, I am a stupid agent...')

        # 2 Only return True if something has been adapted...
        return False




# 2 Implement your own game
class MyGame(Game):

    C_NAME      = 'Matrix'

    def _setup(self, p_mode, p_ada, p_logging):

        # 1 Setup Multi-Player Environment (consisting of 3 OpenAI Gym Cartpole envs)
        self._env   = MultiCartPolePGT(p_num_envs=3, p_logging=p_logging)


        # 2 Setup Multi-Player

        # 2.1 Create empty Multi-Player
        multi_player = MultiPlayer(
            p_name='Human Beings',
            p_ada=p_ada,
            p_logging=p_logging
        )

        # 2.2 Add Single-Player #1 with own policy (controlling sub-environment #1)
        ss_ids = self._env.get_state_space().get_dim_ids()
        as_ids = self._env.get_action_space().get_dim_ids()
        multi_player.add_player(
            p_player=Player(
                p_policy=MyPolicy(
                    p_observation_space=self._env.get_state_space().spawn([ss_ids[0],ss_ids[1],ss_ids[2],ss_ids[3]]),
                    p_action_space=self._env.get_action_space().spawn([as_ids[0]]),
                    p_buffer_size=1,
                    p_ada=p_ada,
                    p_logging=p_logging
                ),
                p_name='Neo',
                p_id=0,
                p_ada=p_ada,
                p_logging=p_logging
            ),
            p_weight=0.3
        )


        # 2.2 Add Single-Player #2 with own policy (controlling sub-environments #2,#3)
        multi_player.add_player(
            p_player=Player(
                p_policy=MyPolicy(
                    p_observation_space=self._env.get_state_space().spawn([ss_ids[4],ss_ids[5],ss_ids[6],ss_ids[7],ss_ids[8],ss_ids[9],ss_ids[10],ss_ids[11]]),
                    p_action_space=self._env.get_action_space().spawn([as_ids[1],as_ids[2]]),
                    p_buffer_size=1,
                    p_ada=p_ada,
                    p_logging=p_logging
                ),
                p_name='Trinity',
                p_id=1,
                p_ada=p_ada,
                p_logging=p_logging
            ),
            p_weight=0.7
        )


        # 2.3 Return multi-player as adaptive model
        return multi_player




# 3 Create game and run some cycles

if __name__ == "__main__":
    # 3.1 Parameters for demo mode
    cycle_limit = 200
    logging     = Log.C_LOG_WE
    visualize   = True
    path        = str(Path.home())
 
else:
    # 3.2 Parameters for internal unit test
    cycle_limit = 10
    logging     = Log.C_LOG_NOTHING
    visualize   = False
    path        = None


# 3.3 Create and run training object
training = GTTraining(
        p_game_cls=MyGame,
        p_cycle_limit=cycle_limit,
        p_path=path,
        p_visualize=visualize,
        p_logging=logging )

training.run()