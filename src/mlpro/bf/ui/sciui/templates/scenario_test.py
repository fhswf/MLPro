## -------------------------------------------------------------------------------------------------
## -- Project : SciUI - Scientific User Interface
## -- Package : sciui.templates
## -- Module  : scenario_test
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2021-06-16  1.0.0     DA       Creation
## -- 2021-09-11  1.0.0     MRD      Change Header information to match our new library name
## -------------------------------------------------------------------------------------------------

"""
Ver. 1.0.0 (2021-06-16)

Elementry functional test scenario for SciUI project. Can be executed directly...
"""



from mlpro.bf.ui.sciui.framework import *



## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class SciUIScenarioTest(SciUIScenario): 

    C_NAME      = 'Test Scenario'
    C_VERSION   = '1.0.0'
    C_VISIBLE   = True
    C_RELEASED  = True
    




if (__name__ == '__main__'): 
    from mlpro.bf.ui.sciui.main import SciUI
    SciUI()

