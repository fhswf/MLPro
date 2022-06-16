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
    "rl_001": "mlpro.rl.examples.howto_rl_001_types_of_reward",
    "rl_002": "mlpro.rl.examples.howto_rl_002_run_agent_with_own_policy_with_gym_environment",
    "rl_003": "mlpro.rl.examples.howto_rl_003_train_agent_with_own_policy_on_gym_environment",
    "rl_004": "mlpro.rl.examples.howto_rl_004_run_multi_agent_with_own_policy_in_multicartpole_environment",
    "rl_005": "mlpro.rl.examples.howto_rl_005_train_multi_agent_with_own_policy_on_multicartpole_nvironment",
    "rl_006": "mlpro.rl.examples.howto_rl_006_run_own_agents_with_petting_zoo_environment",
    "rl_007": "mlpro.rl.examples.howto_rl_007_train_wrapped_SB3_policy",
    "rl_008": "mlpro.rl.examples.howto_rl_008_wrap_mlpro_environment_to_gym_environment",
    "rl_009": "mlpro.rl.examples.howto_rl_009_wrap_mlpro_environment_to_pettingzoo_environment",
    # "rl_010": "mlpro.rl.examples.howto_rl_010_train_ur5_environment_with_wrapped_sb3_policy",
    # "rl_011": "mlpro.rl.examples.howto_rl_011_train_ur5_environment_with_wrapped_sb3_policy",
    "rl_012": "mlpro.rl.examples.howto_rl_012_train_wrapped_SB3_policy_on_robothtm_environment",
    "rl_013": "mlpro.rl.examples.howto_rl_013_model_based_reinforcement_learning",
    "rl_014": "mlpro.rl.examples.howto_rl_014_advanced_training_with_stagnation_detection",
    "rl_015": "mlpro.rl.examples.howto_rl_015_train_wrapped_sb3_policy_with_stagnation_detection",
    "rl_016": "mlpro.rl.examples.howto_rl_016_comparison_native_vs_wrapped_sb3_policy",
    "rl_017": "mlpro.rl.examples.howto_rl_017_comparison_native_vs_wrapped_sb3_policy_off_policy",
    # "rl_018": "mlpro.rl.examples.howto_rl_018_train_wrapped_sb3_policy_on_multigeo_environment",
    "rl_019": "mlpro.rl.examples.howto_rl_019_train_and_reload_single_agent",
    "rl_021": "mlpro.rl.examples.howto_rl_021_train_wrapped_sb3_policy_on_doublependulum",

# Game Theory:
    "gt_001": "mlpro.gt.examples.howto_gt_001_run_multi_player_with_own_policy_in_multicartpole_game_board",
    "gt_002": "mlpro.gt.examples.howto_gt_002_train_own_multi_player_with_multicartpole_game_board"
}



@pytest.mark.parametrize("cls", list(howto_list.keys()))
def test_howto(cls):
    importlib.import_module(howto_list[cls])

