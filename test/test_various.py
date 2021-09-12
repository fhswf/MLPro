## -------------------------------------------------------------------------------------------------
## -- Project : FH-SWF Automation Technology - Common Code Base (CCB)
## -- Package : mlpro
## -- Module  : test_various
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


import unittest
from mlpro.bf.various import Log



## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------

class MyLog(Log):
    C_TYPE      = 'Test class'
    C_NAME      = 'MyLog'

    def __init__(self): 
        Log.__init__(self)



## -------------------------------------------------------------------------------------------------

## -------------------------------------------------------------------------------------------------

class TestVarious(unittest.TestCase):
    """
    Unit tests for module various.py.
    """

## -------------------------------------------------------------------------------------------------

    def test_logging(self):
        """
        Method description.
        """

        lo_log = MyLog()
        self.assertTrue(lo_log.logging)
        lo_log.log(Log.C_LOG_TYPE_I, 'Hello World!')
        lo_log.switch_logging(False)
        self.assertFalse(lo_log.logging)
        lo_log.log(Log.C_LOG_TYPE_I, 'Hello World2!')

       



if __name__ == '__main__': unittest.main()
