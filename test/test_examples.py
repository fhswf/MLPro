## -------------------------------------------------------------------------------------------------
## -- Project : FH-SWF Automation Technology - Common Code Base (CCB)
## -- Package : mlpro
## -- Module  : test_example
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2021-10-06  1.0.0     MRD      Creation
## -- 2021-10-06  1.0.0     MRD      Release First Version
## -- 2021-12-12  1.0.1     DA       Howto 17 added
## -- 2021-12-20  1.0.2     DA       Howto 08 disabled
## -- 2022-02-28  1.0.3     SY       Howto 06, 07 of basic functions are added
## -- 2022-02-28  1.0.4     SY       Howto 07 of basic functions is disabled
## -- 2022-05-29  1.1.0     DA       Update howto list after refactoring of all howto files
## -------------------------------------------------------------------------------------------------

"""
Ver. 1.1.0 (2022-05-29)

Unit test for all examples available.
"""

import pytest
import importlib


howto_list = {

# Basic Functions:
    "bf_001": "mlpro.bf.examples.howto_bf_001_logging",
    "bf_002": "mlpro.bf.examples.howto_bf_002_timer",
    "bf_003": "mlpro.bf.examples.howto_bf_003_spaces_and_elements",
    "bf_004": "mlpro.bf.examples.howto_bf_004_store_plot_and_save_variables",
    "bf_005": "mlpro.bf.examples.howto_bf_005_hyperparameters",
    "bf_006": "mlpro.bf.examples.howto_bf_006_buffers",
    "bf_007": "mlpro.bf.examples.howto_bf_007_hyperparameter_tuning_using_hyperopt",
    "bf_008": "mlpro.bf.examples.howto_bf_008_hyperparameter_tuning_using_optuna",
    "bf_009": "mlpro.bf.examples.howto_bf_009_sciui_reuse_of_interactive_2d_3d_input_space",
    "bf_010": "mlpro.bf.examples.howto_bf_010_sciui_reinforcement_learning_cockpit",

# Reinforcement Learning:

# Game Theory:
    "gt_001": "mlpro.gt.examples.howto_gt_001_run_multi_player_with_own_policy_in_multicartpole_game_board",
    "gt_002": "mlpro.gt.examples.howto_gt_002_train_own_multi_player_with_multicartpole_game_board"
}



@pytest.mark.parametrize("cls", list(howto_list.keys()))
def test_howto(cls):
    importlib.import_module(howto_list[cls])

