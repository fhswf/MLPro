## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - A Synoptic Framework for Standardized Machine Learning Tasks
## -- Package : mlpro
## -- Module  : test_example.py
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
## -- 2022-06-21  1.1.1     SY       Update howto 20 and 21 RL
## -- 2022-09-13  1.1.2     SY       Add howto 22 RL and 03 GT
## -- 2022-10-06  1.1.3     SY       Add howto 23
## -- 2022-10-08  1.1.4     SY       Howto bf 009 and 010 are switched to bf uui 01 and bf uui 02
## -- 2022-10-12  1.2.0     DA       Incorporation of refactored bf howto files
## -- 2022-10-13  1.2.1     DA       Removed howto bf mt 001 due of it's multiprocessing parts
## -- 2022-10-14  1.3.0     SY       Incorporation of refactored bf howto files (RL/GT)
## -- 2022-10-19  1.3.1     DA       Renamed howtos rl_att_001, rl_att_002
## -- 2022-11-07  1.3.2     DA       Reactivated howto bf_streams_011
## -- 2022-11-10  1.3.3     DA       Renamed howtos bf_streams*
## -- 2022-11-22  1.3.4     DA       Removed howto_bf_streams_051 due to delay caused by OpenML
## -- 2022-12-05  1.4.0     DA       Added howto bf_systems_001
## -- 2022-12-09  1.4.1     DA       Temporarily removed howto rl_wp_003 due to problems with pettingzoo
## -- 2022-12-14  1.5.0     DA       Added howtos bf_streams_101, bf_streams_110, bf_streams_111
## -- 2022-12-20  1.6.0     DA       Added howtos bf_streams_112, bf_streams_113
## -- 2022-12-21  1.6.2     SY       - Reactivate howto rl_wp_003
## --                                - Temporarily removed howto rl_agent_005 and rl_wp_001
## -- 2023-01-14  1.6.3     SY       Add howto related to transfer functions
## -- 2023-01-16  1.6.4     SY       Add howto related to unit converters
## -- 2023-01-27  1.6.5     MRD      Add howto related to mujoco
## -- 2023-02-02  1.6.6     DA       Renamed some rl howtos
## -- 2023-02-04  1.6.7     SY       Renaming some bf howtos
## -- 2023-02-15  1.6.8     DA       Renaming and extension of howtos for bf.ml
## -- 2023-02-23  1.6.9     DA       Renamed some rl howtos
## -- 2023-02-23  1.7.0     MRD      new Howto RL Agent 006, Howto BF System 002 and 003
## -- 2023-03-04  1.7.1     DA       Renamed some rl howtos
## -- 2023-03-08  1.7.2     SY       Add Howto RL MB 003
## -- 2023-03-10  1.7.3     SY       Renumbering module
## -- 2023-03-24  1.7.4     DA       Add Howto BF 005
## -------------------------------------------------------------------------------------------------


"""
Ver. 1.7.4 (2023-03-24)

Unit test for all examples available.
"""


import pytest
import runpy


howto_list = {

# MLPro-BF - Basic Functions:

    # bf.various
    "bf_001": "mlpro.bf.examples.howto_bf_001_logging",
    "bf_002": "mlpro.bf.examples.howto_bf_002_timer",
    "bf_003": "mlpro.bf.examples.howto_bf_003_store_plot_and_save_variables",
    "bf_004": "mlpro.bf.examples.howto_bf_004_buffers",
    "bf_005": "mlpro.bf.examples.howto_bf_005_persistence",

    # bf.eh
    "bf_eh_001": "mlpro.bf.examples.howto_bf_eh_001_event_handling",

    # bf.mt
    # "bf_mt_001": "mlpro.bf.examples.howto_bf_mt_001_parallel_algorithms",
    "bf_mt_002": "mlpro.bf.examples.howto_bf_mt_002_tasks_and_workflows",

    # bf.math
    "bf_math_001": "mlpro.bf.examples.howto_bf_math_001_spaces_and_elements",
    "bf_math_010": "mlpro.bf.examples.howto_bf_math_010_normalizers",

    # bf.ml
    "bf_ml_001": "mlpro.bf.examples.howto_bf_ml_001_adaptive_model",
    "bf_ml_010": "mlpro.bf.examples.howto_bf_ml_010_hyperparameters",

    # bf.streams
    "bf_streams_001" : "mlpro.bf.examples.howto_bf_streams_001_accessing_native_data_from_mlpro",
    # "bf_streams_051": "mlpro.bf.examples.howto_bf_streams_051_accessing_data_from_openml",
    "bf_streams_052": "mlpro.bf.examples.howto_bf_streams_052_accessing_data_from_scikitlearn",
    "bf_streams_053": "mlpro.bf.examples.howto_bf_streams_053_accessing_data_from_river",
    "bf_streams_101": "mlpro.bf.examples.howto_bf_streams_101_basics",
    "bf_streams_110": "mlpro.bf.examples.howto_bf_streams_110_stream_task_window",
    "bf_streams_111": "mlpro.bf.examples.howto_bf_streams_111_stream_task_rearranger_2d",
    "bf_streams_112": "mlpro.bf.examples.howto_bf_streams_112_stream_task_rearranger_3d",
    "bf_streams_113": "mlpro.bf.examples.howto_bf_streams_113_stream_task_rearranger_nd",
    "bf_streams_114": "mlpro.bf.examples.howto_bf_streams_114_stream_task_deriver",

    # bf.systems
    "bf_systems_001": "mlpro.bf.examples.howto_bf_systems_001_systems_controllers_actuators_sensors",
    "bf_systems_002": "mlpro.bf.examples.howto_bf_systems_002_doublependulum_systems_wrapped_with_mujoco",
    "bf_systems_003": "mlpro.bf.examples.howto_bf_systems_003_cartpole_continuous_systems_wrapped_with_mujoco",

    # bf.physics
    "bf_physics_001": "mlpro.bf.examples.howto_bf_physics_001_set_up_transfer_functions",
    "bf_physics_002": "mlpro.bf.examples.howto_bf_physics_002_unit_converter",


# MLPro-RL - Reinforcement Learning:
    "rl_001": "mlpro.rl.examples.howto_rl_001_reward",

    "rl_att_001": "mlpro.rl.examples.howto_rl_att_001_train_and_reload_single_agent_gym_sd",
    "rl_att_002": "mlpro.rl.examples.howto_rl_att_002_train_and_reload_single_agent_mujoco_sd_cartpole_discrete",
    "rl_att_003": "mlpro.rl.examples.howto_rl_att_003_train_and_reload_single_agent_mujoco_sd_cartpole_continuous",

    "rl_agent_001": "mlpro.rl.examples.howto_rl_agent_001_run_agent_with_own_policy_on_gym_environment",
    "rl_agent_002": "mlpro.rl.examples.howto_rl_agent_002_train_agent_with_own_policy_on_gym_environment",
    "rl_agent_003": "mlpro.rl.examples.howto_rl_agent_003_run_multiagent_with_own_policy_on_multicartpole_environment",
    "rl_agent_004": "mlpro.rl.examples.howto_rl_agent_004_train_multiagent_with_own_policy_on_multicartpole_environment",
    "rl_agent_011": "mlpro.rl.examples.howto_rl_agent_011_train_and_reload_single_agent_gym",
    "rl_agent_021": "mlpro.rl.examples.howto_rl_agent_021_train_and_reload_single_agent_mujoco_cartpole_discrete",
    "rl_agent_022": "mlpro.rl.examples.howto_rl_agent_022_train_and_reload_single_agent_mujoco_cartpole_continuous",

    "rl_env_001": "mlpro.rl.examples.howto_rl_env_001_train_agent_with_SB3_policy_on_robothtm_environment",
    "rl_env_003": "mlpro.rl.examples.howto_rl_env_003_run_agent_with_random_actions_on_double_pendulum_environment",
    "rl_env_005": "mlpro.rl.examples.howto_rl_env_005_run_agent_with_random_policy_on_double_pendulum_mujoco_environment",

    "rl_ht_001": "mlpro.rl.examples.howto_rl_ht_001_hyperopt",
    "rl_ht_002": "mlpro.rl.examples.howto_rl_ht_002_optuna",

    "rl_mb_001": "mlpro.rl.examples.howto_rl_mb_001_train_and_reload_model_based_agent_gym",
    "rl_mb_002": "mlpro.rl.examples.howto_rl_mb_002_grid_world_environment",
    "rl_mb_003": "mlpro.rl.examples.howto_rl_mb_003_robothtm_environment",

    "rl_ui_001": "mlpro.rl.examples.howto_rl_ui_001_reinforcement_learning_cockpit",

    "rl_wp_001": "mlpro.rl.examples.howto_rl_wp_001_mlpro_environment_to_gym_environment",
    "rl_wp_002": "mlpro.rl.examples.howto_rl_wp_002_mlpro_environment_to_petting_zoo_environment",
    "rl_wp_003": "mlpro.rl.examples.howto_rl_wp_003_run_multiagent_with_own_policy_on_petting_zoo_environment",
    "rl_wp_004": "mlpro.rl.examples.howto_rl_wp_004_train_agent_with_sb3_policy",
    "rl_wp_005": "mlpro.rl.examples.howto_rl_wp_005_validation_wrapped_sb3_on_policy",
    "rl_wp_006": "mlpro.rl.examples.howto_rl_wp_006_validation_wrapped_sb3_off_policy",


# MLPro-GT - Game Theory:
    "gt_dp_001": "mlpro.gt.examples.howto_gt_dp_001_run_multi_player_with_own_policy_on_multicartpole_game_board",
    "gt_dp_002": "mlpro.gt.examples.howto_gt_dp_002_train_own_multi_player_on_multicartpole_game_board",
}



@pytest.mark.parametrize("cls", list(howto_list.keys()))
def test_howto(cls):
    runpy.run_path("src/"+howto_list[cls].replace(".","/")+".py")

