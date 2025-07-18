## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - The integrative middleware framework for standardized machine learning
## -- Package : mlpro.bf.examples
## -- Module  : howto_bf_math_035_geometry_crosshair_1d.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2025-06-25  1.0.0     DA       Creation
## -------------------------------------------------------------------------------------------------

"""
Ver. 1.0.0 (2025-06-25)

This module demonstrates the functionality of class bf.math.geometry.Crosshair in a 1D plot.

You will learn:

1) How to instantiate and update a Crosshair object

2) How to get details from a Crosshair object

3) How to visualize a Crosshair object

"""


from datetime import datetime, timedelta
from math import sin, cos, pi
from time import sleep
import random

import numpy as np

from mlpro.bf import Log, PlotSettings
from mlpro.bf.math.geometry import Crosshair




# 1 Preparation of demo/unit test mode
if __name__ == '__main__':
    # 1.1 Parameters for demo mode
    cycles      = 200
    visualize   = True
    logging     = Log.C_LOG_ALL
  
else:
    # 1.2 Parameters for internal unit test
    cycles      = 5
    visualize   = False
    logging      = Log.C_LOG_NOTHING



# 2 Instantiate a log object
my_log = Log( p_logging = False )
my_log.C_TYPE = 'Demo'
my_log.C_NAME = 'Crosshair'
my_log.switch_logging(p_logging=logging)



# 3 Instantiate a Crosshair object
my_crosshair       = Crosshair( p_name='Crosshair', p_visualize=visualize )
my_crosshair.color = 'blue'
pos                = np.zeros(1)

if __name__ == '__main__':
    my_crosshair.init_plot( p_plot_settings=PlotSettings( p_view = PlotSettings.C_VIEW_ND ))
    my_crosshair.get_plot_settings().axes.set_ylim(0,100)




# 4 Update of crosshair position on a circular path
for i in range(cycles):
    
    if i < 100: 
        pos_step = 1
    else:
        pos_step = -1

    my_crosshair.set( p_value = pos )
    my_crosshair.update_plot()

    pos += pos_step

    if visualize: sleep(0.01)


my_crosshair.remove_plot()


if __name__ == '__main__':
    input('Press ENTER to end the demo...')
