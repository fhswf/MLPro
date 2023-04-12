## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - A Synoptic Framework for Standardized Machine Learning Tasks
## -- Package : mlpro.gt.examples
## -- Module  : howto_gt_dg_004_train_own_multi_player_in_stackelberg_games.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2023-04-12  0.0.0     SY       Creation
## -- 2023-04-12  1.0.0     SY       Release of first version
## -------------------------------------------------------------------------------------------------

"""
Ver. 1.0.0 (2023-04-12)
 
This module shows how to train an own multi-player in stackelberg games.

You will learn:
    
1) How to set up your own players' policies

2) How to set up your own game in dynamic stackelberg games, including players and game board interaction

3) How to assign a player with a role as leader or follower
    
4) How to run the GT training and train your own players
    
"""


from mlpro.rl import *
from mlpro.gt.dynamicgames import *
from mlpro.rl.models import Reward
from mlpro.gt.pool.boards.bglp import BGLP_GT
import random
import numpy as np
from pathlib import Path





# 1 Implement your own player policy
class MyPolicy(Policy):

    C_NAME      = 'MyPolicy'

    def compute_action(self, p_state: State, p_action_leaders=False) -> Action:
        # 1 Create a numpy array for your action values 
        my_action_values = np.zeros(self._action_space.get_num_dim())

        # 2 Computing action values is up to you...
        for d in range(self._action_space.get_num_dim()):
            my_action_values[d] = random.random() 

        # 3 Return an action object with your values
        return Action(self._id, self._action_space, my_action_values)


    def _adapt(self, **p_args) -> bool:
        # 1 Adapting the internal policy is up to you...
        self.log(self.C_LOG_TYPE_I, 'Sorry, I am a stupid agent...')

        # 2 Only return True if something has been adapted...
        return False
    



# 2 Implement your own game
class MyGame(GTGame_DG):

    C_NAME      = 'Matrix'

    def _setup(self, p_mode, p_ada: bool, p_visualize: bool, p_logging) -> Model:

        # 1 Setup Multi-Player Environment (consisting of 3 OpenAI Gym Cartpole envs)
        self._env   = BGLP_GT(p_logging=p_logging)


        # 2 Setup Multi-Player

        # 2.1 Create empty Multi-Player
        multi_player = GTMultiPlayer_SG(
            p_name='BGLP Players with Random Policies',
            p_ada=p_ada,
            p_visualize=p_visualize,
            p_logging=p_logging
        )

        # 2.2 Add Single-Players with own policy
        ss_ids = self._env.get_state_space().get_dim_ids()
        as_ids = self._env.get_action_space().get_dim_ids()

        # Player 1
        multi_player.add_player(
            p_player=GTPlayer_SG(
                p_policy=MyPolicy(
                    p_observation_space=self._env.get_state_space().spawn([ss_ids[0],ss_ids[1]]),
                    p_action_space=self._env.get_action_space().spawn([as_ids[0]]),
                    p_buffer_size=1,
                    p_ada=p_ada,
                    p_visualize=p_visualize,
                    p_logging=p_logging
                ),
                p_name='BELT_CONVEYOR_A (Leader)',
                p_id=0,
                p_ada=p_ada,
                p_visualize=p_visualize,
                p_logging=p_logging,
                p_role=GTPlayer_SG.C_PLAYER_LEADER
            ),
            p_weight=1.0
        )
        
        # Player 2
        multi_player.add_player(
            p_player=GTPlayer_SG(
                p_policy=MyPolicy(
                    p_observation_space=self._env.get_state_space().spawn([ss_ids[1],ss_ids[2]]),
                    p_action_space=self._env.get_action_space().spawn([as_ids[1]]),
                    p_buffer_size=1,
                    p_ada=p_ada,
                    p_visualize=p_visualize,
                    p_logging=p_logging
                ),
                p_name='VACUUM_PUMP_B (Follower)',
                p_id=1,
                p_ada=p_ada,
                p_visualize=p_visualize,
                p_logging=p_logging,
                p_role=GTPlayer_SG.C_PLAYER_FOLLOWER
            ),
            p_weight=1.0
        )
        
        # Player 3
        multi_player.add_player(
            p_player=GTPlayer_SG(
                p_policy=MyPolicy(
                    p_observation_space=self._env.get_state_space().spawn([ss_ids[2],ss_ids[3]]),
                    p_action_space=self._env.get_action_space().spawn([as_ids[2]]),
                    p_buffer_size=1,
                    p_ada=p_ada,
                    p_visualize=p_visualize,
                    p_logging=p_logging
                ),
                p_name='VIBRATORY_CONVEYOR_B (Follower)',
                p_id=2,
                p_ada=p_ada,
                p_visualize=p_visualize,
                p_logging=p_logging,
                p_role=GTPlayer_SG.C_PLAYER_FOLLOWER
            ),
            p_weight=1.0
        )
        
        # Player 4
        multi_player.add_player(
            p_player=GTPlayer_SG(
                p_policy=MyPolicy(
                    p_observation_space=self._env.get_state_space().spawn([ss_ids[3],ss_ids[4]]),
                    p_action_space=self._env.get_action_space().spawn([as_ids[3]]),
                    p_buffer_size=1,
                    p_ada=p_ada,
                    p_visualize=p_visualize,
                    p_logging=p_logging
                ),
                p_name='VACUUM_PUMP_C (Follower)',
                p_id=3,
                p_ada=p_ada,
                p_visualize=p_visualize,
                p_logging=p_logging,
                p_role=GTPlayer_SG.C_PLAYER_FOLLOWER
            ),
            p_weight=1.0
        )
        
        # Player 5
        multi_player.add_player(
            p_player=GTPlayer_SG(
                p_policy=MyPolicy(
                    p_observation_space=self._env.get_state_space().spawn([ss_ids[4],ss_ids[5]]),
                    p_action_space=self._env.get_action_space().spawn([as_ids[4]]),
                    p_buffer_size=1,
                    p_ada=p_ada,
                    p_visualize=p_visualize,
                    p_logging=p_logging
                ),
                p_name='ROTARY_FEEDER_C (Leader)',
                p_id=4,
                p_ada=p_ada,
                p_visualize=p_visualize,
                p_logging=p_logging,
                p_role=GTPlayer_SG.C_PLAYER_LEADER
            ),
            p_weight=1.0
        )

        # 2.3 Return multi-player as adaptive model
        return multi_player




# 3 Create game and run some cycles

if __name__ == "__main__":
    # 3.1 Parameters for demo mode
    cycle_limit = 200
    logging     = Log.C_LOG_ALL
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