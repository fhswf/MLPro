## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - The integrative middleware framework for standardized machine learning
## -- Package : mlpro.gt.examples
## -- Module  : howto_gt_dg_003_train_multi_player_with_SbPG_on_BGLP.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2025-04-09  0.0.0     SY       Creation
## -- 2025-04-09  1.0.0     SY       Release of first version
## -------------------------------------------------------------------------------------------------

"""
Ver. 1.0.0 (2025-04-09)
 
This module shows how to train a multi-player in SbPG on the BGLP game board.

You will learn:
    
1) How to set up your an SbPG
    
2) How to run the SbPG training and train the multi-player
    
"""


from mlpro.rl import *
from mlpro.gt import *
from mlpro.gt.dynamicgames.potential import *
from mlpro.rl.models import Reward
from mlpro.rl.pool.envs.bglp import BGLP
from mlpro.gt.pool.policies.sbpg import SbPG
import random
import numpy as np
from pathlib import Path





# 1 Setting up the utility function in the BGLP
class MyBGLP(BGLP):

    C_NAME          = 'MyBGLP'

    def __init__(
            self, 
            p_logging=Log.C_LOG_ALL,
            t_set=10.0,
            demand=0.11,
            lr_margin=1.0,
            lr_demand=4.0,
            lr_power=0.0010
            ):
        BGLP.__init__(
            self,
            p_logging=p_logging,
            t_set=t_set,
            demand=demand,
            lr_margin=lr_margin,
            lr_demand=lr_demand,
            lr_power=lr_power
            )
                          

    def calc_reward(self):
        
        for actnum in range(len(self.acts)):
            acts                    = self.acts[actnum]
            self.reward[actnum]     = 1/(1+self.lr_margin*self.margin_t[actnum])+1/(1+self.lr_power*self.power_t[actnum]/(acts.power_max/1000.0))
            if actnum == len(self.acts)-1:
                self.reward[actnum] += 1/(1-self.lr_demand*self.demand_t[-1])
            else:
                self.reward[actnum] += 1/(1+self.lr_margin*self.margin_t[actnum+1])

        return self.reward[:]






# 2 Setting up an SbPG for the BGLP
class SbPG_Scenario(Game):

    C_NAME = 'SbPG_Scenario'
    

    def _setup(self, p_mode, p_ada, p_visualize, p_logging):
        self._env       = MyBGLP(p_logging=p_logging)
        multi_player    = MultiPlayer(
            p_name='SbPG Players',
            p_ada=p_ada,
            p_logging=p_logging,
            p_visualize=p_visualize
            )
        state_space     = self._env.get_state_space()
        action_space    = self._env.get_action_space()
        
        ## -- Select one learning algorithm -- ##
        # _algo = SbPG.ALG_SbPG_BR
        _algo = SbPG.ALG_SbPG_GB
        # _algo = SbPG.ALG_SbPG_GB_MOM
        
        # Player 1
        _name         = 'BELT_CONVEYOR_A'
        _id           = 0
        _ospace       = state_space.spawn([state_space.get_dim_ids()[0],state_space.get_dim_ids()[1]])
        _aspace       = action_space.spawn([action_space.get_dim_ids()[0]])
        _policy       = SbPG(
            p_observation_space=_ospace,
            p_action_space=_aspace,
            p_buffer_size=1,
            p_ada=1, 
            p_logging=p_logging,
            p_algo=_algo
            )
        multi_player.add_player(
            p_player=Player(
                p_policy=_policy,
                p_envmodel=None,
                p_name=_name,
                p_id=_id,
                p_ada=True,
                p_logging=p_logging
                ),
            p_weight=1.0
            )
        
        
        # Player 2
        _name         = 'VACUUM_PUMP_B'
        _id           = 1
        _ospace       = state_space.spawn([state_space.get_dim_ids()[1],state_space.get_dim_ids()[2]])
        _aspace       = action_space.spawn([action_space.get_dim_ids()[1]])
        _policy       = SbPG(
            p_observation_space=_ospace,
            p_action_space=_aspace,
            p_buffer_size=1,
            p_ada=1, 
            p_logging=p_logging,
            p_algo=_algo
            )
        multi_player.add_player(
            p_player=Player(
                p_policy=_policy,
                p_envmodel=None,
                p_name=_name,
                p_id=_id,
                p_ada=True,
                p_logging=p_logging
                ),
            p_weight=1.0
            )
        
        
        # Player 3
        _name         = 'VIBRATORY_CONVEYOR_B'
        _id           = 2
        _ospace       = state_space.spawn([state_space.get_dim_ids()[2],state_space.get_dim_ids()[3]])
        _aspace       = action_space.spawn([action_space.get_dim_ids()[2]])
        _policy       = SbPG(
            p_observation_space=_ospace,
            p_action_space=_aspace,
            p_buffer_size=1,
            p_ada=1, 
            p_logging=p_logging,
            p_algo=_algo
            )
        multi_player.add_player(
            p_player=Player(
                p_policy=_policy,
                p_envmodel=None,
                p_name=_name,
                p_id=_id,
                p_ada=True,
                p_logging=p_logging
                ),
            p_weight=1.0
            )
        
        
        # Player 4
        _name         = 'VACUUM_PUMP_C'
        _id           = 3
        _ospace       = state_space.spawn([state_space.get_dim_ids()[3],state_space.get_dim_ids()[4]])
        _aspace       = action_space.spawn([action_space.get_dim_ids()[3]])
        _policy       = SbPG(
            p_observation_space=_ospace,
            p_action_space=_aspace,
            p_buffer_size=1,
            p_ada=1, 
            p_logging=p_logging,
            p_algo=_algo
            )
        multi_player.add_player(
            p_player=Player(
                p_policy=_policy,
                p_envmodel=None,
                p_name=_name,
                p_id=_id,
                p_ada=True,
                p_logging=p_logging
                ),
            p_weight=1.0
            )
        
        
        # Player 5
        _name         = 'ROTARY_FEEDER_C'
        _id           = 4
        _ospace       = state_space.spawn([state_space.get_dim_ids()[4],state_space.get_dim_ids()[5]])
        _aspace       = action_space.spawn([action_space.get_dim_ids()[4]])
        _policy       = SbPG(
            p_observation_space=_ospace,
            p_action_space=_aspace,
            p_buffer_size=1,
            p_ada=1, 
            p_logging=p_logging,
            p_algo=_algo
            )
        multi_player.add_player(
            p_player=Player(
                p_policy=_policy,
                p_envmodel=None,
                p_name=_name,
                p_id=_id,
                p_ada=True,
                p_logging=p_logging
                ),
            p_weight=1.0
            )
        
        return multi_player





if __name__ == "__main__":
    logging         = Log.C_LOG_WE
    visualize       = False
    dest_path       = str(Path.home())
    cycle_limit     = 100000
    cycle_per_ep    = 10000
    eval_freq       = 0
    eval_grp_size   = 0
    adapt_limit     = 0
    stagnant_limit  = 0
    score_ma_hor    = 0
else:
    logging         = Log.C_LOG_NOTHING
    visualize       = False
    dest_path       = str(Path.home())
    cycle_limit     = 10
    cycle_per_ep    = 10
    eval_freq       = 0
    eval_grp_size   = 0
    adapt_limit     = 0
    stagnant_limit  = 0
    score_ma_hor    = 0
    
training = RLTraining(
    p_scenario_cls=SbPG_Scenario,
    p_cycle_limit=cycle_limit,
    p_cycles_per_epi_limit=cycle_per_ep,
    p_eval_frequency=eval_freq,
    p_eval_grp_size=eval_grp_size,
    p_adaptation_limit=adapt_limit,
    p_stagnation_limit=stagnant_limit,
    p_score_ma_horizon=score_ma_hor,
    p_collect_states=True,
    p_collect_actions=True,
    p_collect_rewards=True,
    p_collect_training=True,
    p_visualize=visualize,
    p_path=dest_path,
    p_logging=logging
    )

training.run()
if __name__ == "__main__":
    training._scenario.get_env().data_storing.save_data(training._root_path, 'bglp')
