## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - A Synoptic Framework for Standardized Machine Learning Tasks
## -- Package : mlpro.oa.sciui
## -- Module  : scenario_oa1.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2021-06-20  0.0.0     DA       Creation
## -- 2021-07-03  1.0.0     DA       Release of first version
## -- 2022-01-06  1.1.0     DA       Integration in mlpro
## -- 2022-06-04  1.1.1     DA       Refactoring
## -------------------------------------------------------------------------------------------------

"""
Ver. 1.1.1 (2022-06-04)

Demo scenarios for SciUI framework that shows the reuse of the interactive 2D/3D input space class.
Can be executed directly...
"""



from mlpro.bf.ui.sciui.framework import *
from mlpro.oa.sciui.iis import InteractiveInputSpace
from mlpro.bf.math import *




## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class DSM2D(SciUIScenario):

    C_NAME          = 'Data Stream Mining - Processing of 2D Streams'
    C_VERSION       = '0.0.0'
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
class DSM3D(SciUIScenario):

    C_NAME          = 'Data Stream Mining - Processing of 3D Streams'
    C_VERSION       = '0.0.0'
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
                                                      p_boundaries=[-25,25]))
        self.shared_db.iis_ispace.add_dim( Dimension( p_name_short='x3', 
                                                      p_description='', 
                                                      p_name_latex='x_3', 
                                                      p_unit='m/s^2', 
                                                      p_unit_latex='\\frac{m}{s^2}', 
                                                      p_boundaries=[-15,15]) )

        # 2 Build scenario structure
        self.add_component(InteractiveInputSpace(self.shared_db, p_row=0, p_col=0, p_padx=5, p_logging=self._level))



        

if (__name__ == '__main__'): 
    from mlpro.bf.ui.sciui.main import SciUI
    SciUI()
