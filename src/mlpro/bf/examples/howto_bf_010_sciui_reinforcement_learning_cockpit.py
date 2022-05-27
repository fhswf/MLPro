## -------------------------------------------------------------------------------------------------
## -- Project : FH-SWF Automation Technology - Common Code Base (CCB)
## -- Package : mlpro
## -- Module  : howto_bf_010_sciui_reinforcement_learning_cockpit.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2021-07-27  0.0.0     SY       Creation
## -- 2021-10-07  1.0.0     SY       Release of first draft
## -------------------------------------------------------------------------------------------------

"""
Ver. 1.0.0 (2021-10-07)

SciUI template for a Reinforcement Learning simulation.
"""



from mlpro.bf.ui.sciui.framework import *
from mlpro.bf.math import *
from mlpro.rl.sciui_rl import RLInteractiveUI




## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class SciUI_RL(SciUIScenario):

    C_NAME          = 'Reinforcement Learning'
    C_VERSION       = '1.0.0'
    C_RELEASED      = True
    C_VISIBLE       = True
    
## -------------------------------------------------------------------------------------------------
    def init_component(self):
        super().init_component()
        RLInteractiveUI.enrich_shared_db(self.shared_db)
        
        self.add_component(RLInteractiveUI(p_shared_db=self.shared_db, p_row=0, p_col=0,
                                           p_padx=5, p_logging=self._level, p_refresh_rate=100))
        



if (__name__ == '__main__'): 
    from mlpro.bf.ui.sciui.main import SciUI
    SciUI()