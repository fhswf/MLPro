## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - The integrative middleware framework for standardized machine learning
## -- Package : mlpro.bf.examples
## -- Module  : howto_bf_math_021_geometry_point_3d.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2023-05-06  1.0.0     DA       Creation
## -- 2023-09-25  1.0.1     DA       Bugfix
## -- 2024-04-29  1.1.0     DA       Refactoring
## -- 2024-05-05  1.2.0     DA       Refactoring
## -------------------------------------------------------------------------------------------------

"""
Ver. 1.2.0 (2024-05-05)

This module demonstrates the functionality of class bf.math.geometry.Point in a 3D plot.

You will learn:

1) How to instantiate and update a Point object

2) How to get details from a Point object

3) How to visualize a Point object

"""


from mlpro.bf.various import Log
from mlpro.bf.plot import PlotSettings
from mlpro.bf.math.geometry import Point
from datetime import datetime, timedelta
import numpy as np
from math import sin, cos, pi
from time import sleep
import random




# 1 Preparation of demo/unit test mode
if __name__ == '__main__':
    # 1.1 Parameters for demo mode
    cycles      = 50
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



# 3 Instantiate a Point object
time_stamp = datetime.now()
time_step  = timedelta(0,1,0)
my_point   = Point( p_derivative_order_max=2, p_visualize=visualize )
pos        = np.zeros(3)

if __name__ == '__main__':
    my_point.init_plot( p_plot_settings=PlotSettings( p_view=PlotSettings.C_VIEW_3D))



# 4 Update of point position on a circular path
angle = 0
random.seed(1)

for i in range(200):
    angle_step = 3 + random.random()
    if i < 100: 
        angle += angle_step
    else:
        angle -= angle_step

    pos[0] = cos( angle * pi / 180 )
    pos[1] = sin( angle * pi / 180 )
    my_point.set( p_value = pos, p_time_stamp = time_stamp )

    try:
        vel = my_point.derivatives[1]
        acc = my_point.derivatives[2]
        my_log.log(Log.C_LOG_TYPE_S, 'pos :', pos, ', vel :', vel, ', acc :', acc)
    except:
        pass

    time_stamp += time_step

    if visualize: sleep(0.05)



if __name__ == '__main__':
    input('Press ENTER to end the demo...')
