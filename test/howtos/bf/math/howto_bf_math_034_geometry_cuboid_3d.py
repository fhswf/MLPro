## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - The integrative middleware framework for standardized machine learning
## -- Package : mlpro.bf.examples
## -- Module  : howto_bf_math_034_geometry_cuboid_3d.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2024-06-03  1.0.0     DA       Creation
## -------------------------------------------------------------------------------------------------

"""
Ver. 1.0.0 (2024-06-03)

This module demonstrates the functionality of class bf.math.geometry.Hypercuboid in a 3D plot.

You will learn:

1) How to instantiate and update a Hypercuboid object

2) How to get details from a Hypercuboid object

3) How to visualize a Hypercuboid object

"""

from datetime import datetime, timedelta
from math import sin, cos, pi
from time import sleep
import random

import numpy as np

from mlpro.bf.various import Log
from mlpro.bf.plot import PlotSettings
from mlpro.bf.math.geometry import Hypercuboid





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
my_log.C_NAME = 'Point'
my_log.switch_logging(p_logging=logging)



# 3 Instantiate a cuboid object
time_stamp = datetime.now()
time_step  = timedelta(0,1,0)
my_cuboid  = Hypercuboid( p_name='Cuboid', 
                          p_derivative_order_max=0, 
                          p_value_prev=False, 
                          p_visualize=visualize )
my_cuboid.color = 'green'
my_cuboid.alpha = 0.2

if __name__ == '__main__':
    my_cuboid.init_plot( p_plot_settings=PlotSettings( p_view=PlotSettings.C_VIEW_3D ))



# 4 Update of point position on a circular path
angle = 0
boundaries = np.ndarray( shape=(3,2) )
random.seed(1)

for i in range(cycles):
    if i < 100: 
        angle += 3
    else:
        angle -= 3

    boundaries[0][0] = cos( angle * pi / 180 ) * 50
    boundaries[1][0] = sin( angle * pi / 180 ) * 50
    boundaries[2][0] = -12.5
    boundaries[0][1] = boundaries[0][0] + 25
    boundaries[1][1] = boundaries[1][0] + 25
    boundaries[2][1] = boundaries[2][0] + 25

    my_cuboid.set( p_value = boundaries, p_time_stamp = time_stamp )
    my_cuboid.update_plot()

    my_log.log( Log.C_LOG_TYPE_I, 'Geometric center: ', my_cuboid.center_geo.value)

    time_stamp += time_step

    if visualize: sleep(0.05)

my_cuboid.remove_plot()


if __name__ == '__main__':
    input('Press ENTER to end the demo...')
