## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - The integrative middleware framework for standardized machine learning
## -- Package : mlpro.bf.examples
## -- Module  : howto_bf_math_037_geometry_crosshair_3d.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2024-10-31  1.0.0     DA       Creation
## -------------------------------------------------------------------------------------------------

"""
Ver. 1.0.0 (2024-10-31)

This module demonstrates the functionality of class bf.math.geometry.Crosshair in a 3D plot.

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
time_stamp         = datetime.now()
time_step          = timedelta(0,1,0)
my_crosshair       = Crosshair( p_name='Crosshair', p_visualize=visualize )
my_crosshair.color = 'blue'
pos                = np.zeros(3)

if __name__ == '__main__':
    my_crosshair.init_plot( p_plot_settings=PlotSettings( p_view=PlotSettings.C_VIEW_3D ))



# 4 Update of crosshair position on a circular path
angle = 0
random.seed(1)

for i in range(cycles):
    angle_step = 3 + random.random()
    if i < 100: 
        angle += angle_step
    else:
        angle -= angle_step

    pos[0] = cos( angle * pi / 180 )
    pos[1] = sin( angle * pi / 180 )
    my_crosshair.set( p_value = pos, p_time_stamp = time_stamp )
    my_crosshair.update_plot()

    time_stamp += time_step

    if visualize: sleep(0.05)

my_crosshair.remove_plot()


if __name__ == '__main__':
    input('Press ENTER to end the demo...')
