## -------------------------------------------------------------------------------------------------
## -- Project : FH-SWF Automation Technology - Common Code Base (CCB)
## -- Package : mlpro
## -- Module  : Howto 01 (SciUI) - Reuse of interactive 2D,3D input space
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2021-06-20  0.0.0     DA       Creation
## -- 2021-07-03  1.0.0     DA       Release of first version
## -- 2021-09-11  1.0.0     MRD      Change Header information to match our new library name
## -------------------------------------------------------------------------------------------------

"""
Ver. 1.0.0 (2021-07-03)

Demo scenarios for SciUI framework that shows the reuse of the interactive 2D/3D input space class.
Can be executed directly...
"""



from mlpro.bf.ui.framework import *
from mlpro.bf.ui.pool.iis import InteractiveInputSpace
from mlpro.bf.math import *




## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class DemoIIS2D(SciUIScenario):

    C_NAME          = 'Demo for interactive 2D Input Space'
    C_VERSION       = '1.0.0'
    C_RELEASED      = True
    C_VISIBLE       = True

## -------------------------------------------------------------------------------------------------
    def init_component(self):
        super().init_component()

        # 1 Add scenario-specific variables to shared db
        InteractiveInputSpace.enrich_shared_db(self.shared_db)
        self.shared_db.iis_ispace.add_dim(Dimension(0, 'x1', '', 'x_1', 'm', 'm', [-5,5]))
        self.shared_db.iis_ispace.add_dim(Dimension(1, 'x2', '', 'x_2', 'm/s', '\\frac{m}{s}', [-25,25]))

        # 2 Build scenario structure
        self.add_component(InteractiveInputSpace(self.shared_db, p_row=0, p_col=0, p_padx=5, p_logging=self._level))





## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class DemoIIS3D(SciUIScenario):

    C_NAME          = 'Demo for interactive 3D Input Space'
    C_VERSION       = '1.0.0'
    C_RELEASED      = True
    C_VISIBLE       = True

## -------------------------------------------------------------------------------------------------
    def init_component(self):
        super().init_component()

        # 1 Add scenario-specific variables to shared db
        InteractiveInputSpace.enrich_shared_db(self.shared_db)
        self.shared_db.iis_ispace.add_dim(Dimension(0, 'x1', '', 'x_1', 'm', 'm', [-5,5]))
        self.shared_db.iis_ispace.add_dim(Dimension(1, 'x2', '', 'x_2', 'm/s', '\\frac{m}{s}', [-25,25]))
        self.shared_db.iis_ispace.add_dim(Dimension(2, 'x3', '', 'x_3', 'm/s^2', '\\frac{m}{s^2}', [-15,15]))

        # 2 Build scenario structure
        self.add_component(InteractiveInputSpace(self.shared_db, p_row=0, p_col=0, p_padx=5, p_logging=self._level))



        

if (__name__ == '__main__'): 
    from mlpro.bf.ui.main import SciUI
    SciUI()
