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
## -------------------------------------------------------------------------------------------------

"""
Ver. 1.0.4 (2022-02-28)

Unit test for all examples available.
"""

import pytest
import importlib

howto_list = {
    "bf_01": "examples.bf.Howto 01 - (Various) Logging",
    "bf_02": "examples.bf.Howto 02 - (Various) Timer",
    "bf_03": "examples.bf.Howto 03 - (Math) Spaces, subspaces and elements",
    "bf_05": "examples.bf.Howto 05 - (ML) Hyperparameters setup",
    "bf_06": "examples.bf.Howto 06 - (Data) Buffer",
    "gt_06": "examples.gt.Howto 06 - (GT) Run multi-player with own policy in multicartpole game board",
    "gt_07": "examples.gt.Howto 07 - (GT) Train own multi-player with multicartpole game board",
    "rl_01": "examples.rl.Howto 01 - (RL) Types of reward",
    "rl_02": "examples.rl.Howto 02 - (RL) Run agent with own policy with gym environment",
    "rl_03": "examples.rl.Howto 03 - (RL) Train agent with own policy on gym environment",
    "rl_04": "examples.rl.Howto 04 - (RL) Run multi-agent with own policy in multicartpole environment",
    "rl_05": "examples.rl.Howto 05 - (RL) Train multi-agent with own policy on multicartpole environment",
    "rl_08": "examples.rl.Howto 08 - (RL) Run own agents with petting zoo environment",
    "rl_10": "examples.rl.Howto 10 - (RL) Train using SB3 Wrapper",
    "rl_11": "examples.rl.Howto 11 - (RL) Wrap mlpro Environment class to gym environment",
    "rl_12": "examples.rl.Howto 12 - (RL) Wrap mlpro Environment class to petting zoo environment",
    "rl_13": "examples.rl.Howto 13 - (RL) Comparison Native and Wrapper SB3 Policy",
    "rl_15": "examples.rl.Howto 15 - (RL) Train Robothtm with SB3 Wrapper",
    "rl_16": "examples.rl.Howto 16 - (RL) Model Based Reinforcement Learning",
    "rl_17": "examples.rl.Howto 17 - (RL) Advanced training with stagnation detection",
    "rl_18": "examples.rl.Howto 18 - (RL) Single Agent with stagnation detection and SB3 Wrapper",
    "sciui_01": "examples.sciui.Howto 01 (SciUI) - Reuse of interactive 2D,3D input space"
}

@pytest.mark.parametrize("cls", list(howto_list.keys()))
def test_howto(cls):
    importlib.import_module(howto_list[cls])

