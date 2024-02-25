## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - A Synoptic Framework for Standardized Machine Learning Tasks
## -- Package : mlpro
## -- Module  : test_various.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2021-04-16  1.0.0     DA       Creation
## -- 2021-09-11  1.0.0     MRD      Change Header information to match our new library name
## -------------------------------------------------------------------------------------------------

"""
Ver. 1.0.0 (2021-04-16)

Unit test classes for various basic functions.
"""


import pytest
from mlpro.bf.various import Log



## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class MyLog(Log):
    C_TYPE      = 'Test class'
    C_NAME      = 'MyLog'

    def __init__(self): 
        Log.__init__(self)


## -------------------------------------------------------------------------------------------------
def test_logging(capsys):
    """
    Method description.
    """

    lo_log = MyLog()
    lo_log.log(Log.C_LOG_TYPE_I, 'Information')
    lo_log.log(Log.C_LOG_TYPE_W, 'Warning')
    lo_log.log(Log.C_LOG_TYPE_E, 'Error')
    captured = capsys.readouterr()
    assert 'Information' in captured.out
    assert 'Warning' in captured.out
    assert 'Error' in captured.out
    lo_log.switch_logging(Log.C_LOG_NOTHING)
    lo_log.log(Log.C_LOG_TYPE_I, 'Information')
    lo_log.log(Log.C_LOG_TYPE_W, 'Warning')
    lo_log.log(Log.C_LOG_TYPE_E, 'Error')
    captured = capsys.readouterr()
    assert 'Information' not in captured.out
    assert 'Warning' not in captured.out
    assert 'Error' not in captured.out
    lo_log.switch_logging(Log.C_LOG_WE)
    lo_log.log(Log.C_LOG_TYPE_I, 'Information')
    lo_log.log(Log.C_LOG_TYPE_W, 'Warning')
    lo_log.log(Log.C_LOG_TYPE_E, 'Error')
    captured = capsys.readouterr()
    assert 'Information' not in captured.out
    assert 'Warning' in captured.out
    assert 'Error' in captured.out
    lo_log.switch_logging(Log.C_LOG_E)
    lo_log.log(Log.C_LOG_TYPE_I, 'Information')
    lo_log.log(Log.C_LOG_TYPE_W, 'Warning')
    lo_log.log(Log.C_LOG_TYPE_E, 'Error')
    captured = capsys.readouterr()
    assert 'Information' not in captured.out
    assert 'Warning' not in captured.out
    assert 'Error' in captured.out
    
        
