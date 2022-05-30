## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - A Synoptic Framework for Standardized Machine Learning Tasks
## -- Package : mlpro
## -- Module  : howto_bf_009_sciui_reuse_of_interactive_2d_3d_input_space.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2021-06-20  0.0.0     DA       Creation
## -- 2021-07-03  1.0.0     DA       Release of first version
## -- 2021-09-11  1.0.0     MRD      Change Header information to match our new library name
## -- 2022-01-06  1.0.1     DA       Corrections
## -- 2022-03-21  1.0.2     SY       Refactoring following class Dimensions update 
## -------------------------------------------------------------------------------------------------

"""
Ver. 1.0.2 (2022-03-21)

Demo scenarios for SciUI framework that shows the reuse of the interactive 2D/3D input space class.
Can be executed directly...
"""



from mlpro.bf.ui.sciui.framework import *
from mlpro.bf.ui.sciui.pool.iis import InteractiveInputSpace
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
        self.shared_db.iis_ispace.add_dim( Dimension( p_name_short='x1',
                                                      p_description='', 
                                                      p_name_latex='x_1', 
                                                      p_unit='m', 
                                                      p_unit_latex='m', 
                                                      p_boundaries=[-5,5]) )
        self.shared_db.iis_ispace.add_dim( Dimension( p_name_short='x2', 
                                                      p_description='', 
                                                      p_name_latex='x_2', 
                                                      p_unit='m/s', 
                                                      p_unit_latex='\\frac{m}{s}', 
                                                      p_boundaries=[-25,25]) )

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
        self.shared_db.iis_ispace.add_dim( Dimension( p_id=0, 
                                                      p_name_short='x1', 
                                                      p_description='', 
                                                      p_name_latex='x_1', 
                                                      p_unit='m', 
                                                      p_unit_latex='m', 
                                                      p_boundaries=[-5,5]) )
        self.shared_db.iis_ispace.add_dim( Dimension( p_id=1, 
                                                      p_name_short='x2', 
                                                      p_description='', 
                                                      p_name_latex='x_2', 
                                                      p_unit='m/s', 
                                                      p_unit_latex='\\frac{m}{s}', 
                                                      p_boundaries=[-25,25]) )
        self.shared_db.iis_ispace.add_dim( Dimension( p_id=2, 
                                                      p_name_short='x3', 
                                                      p_description='', 
                                                      p_name_latex='x_3', 
                                                      p_unit='m/s^2', 
                                                      p_unit_latex='\\frac{m}{s^2}', 
                                                      p_boundaries=[-15,15] ))

        # 2 Build scenario structure
        self.add_component(InteractiveInputSpace(self.shared_db, p_row=0, p_col=0, p_padx=5, p_logging=self._level))



        

if (__name__ == '__main__'): 
    from mlpro.bf.ui.sciui.main import SciUI
    SciUI()
