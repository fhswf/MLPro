## -------------------------------------------------------------------------------------------------
## -- Project : FH-SWF Automation Technology - Common Code Base (CCB)
## -- Package : mlpro
## -- Module  : test_example
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2021-12-14  1.0.0     MRD      Creation
## -- 2021-12-14  1.0.0     MRD      Release First Version
## -------------------------------------------------------------------------------------------------

"""
Ver. 1.0.1 (2021-12-12)

Unit test for all examples available.
"""

import pytest
import importlib

howto_list = {
    "rl_14": "examples.rl.Howto 14 - (RL) Train UR5 with SB3 wrapper"
}

@pytest.mark.parametrize("cls", list(howto_list.keys()))
def test_howto(cls):
    importlib.import_module(howto_list[cls])

