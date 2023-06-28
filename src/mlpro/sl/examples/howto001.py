## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - A Synoptic Framework for Standardized Machine Learning Tasks
## -- Package : mlpro.sl
## -- Module  : models_data.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2023-06-18  0.0.0     LSB      Creation
## -------------------------------------------------------------------------------------------------

"""
Ver. 0.0.0 (2023-06-13)

This module provides dataset classes for supervised learning tasks.
"""



from mlpro.sl.models_dataset import SASDataset
from pathlib import Path
import os
from mlpro.rl.pool.envs.doublependulum import *

path = Path.home()

path_state = str(path) + os.sep + 'Results-2\env_states.csv'
path_action = str(path) + os.sep + 'Results-2\\agent_actions.csv'

dp = DoublePendulumS4()
state_space,action_space = dp.setup_spaces()

state_space = state_space.spawn(state_space.get_dim_ids()[:-1])

print([i.get_name_long() for i in state_space.get_dims()])

mydataset = SASDataset(p_state_fpath=path_state,
                        p_action_fpath=path_action,
                        p_state_space=state_space,
                        p_action_space=action_space)

for i in mydataset:
    print(i)