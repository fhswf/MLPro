## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - A Synoptic Framework for Standardized Machine Learning Tasks
## -- Package : mlpro.bf.examples
## -- Module  : howto_bf_math_021_geometry_point_3d.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2023-05-06  1.0.0     DA       Creation
## -------------------------------------------------------------------------------------------------

"""
Ver. 1.0.0 (2023-05-06)

This module demonstrates the functionality of class bf.math.geometry.Point in a 3D plot.

You will learn:

1) How to instantiate and update a Point object

2) How to get details from a Point object

3) How to visualize a Point object

"""


from mlpro.bf.various import Log
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
    loging      = Log.C_LOG_NOTHING



# 2 Instantiate a log object
my_log = Log( p_logging = False )
my_log.C_TYPE = 'Demo'
my_log.C_NAME = 'Point'
my_log.switch_logging(p_logging=logging)



# 3 Instantiate a Point object
time_stamp = datetime.now()
time_step  = timedelta(0,1,0)
my_point   = Point( p_visualize = visualize )
pos        = np.zeros(3)



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
    my_point.set_values( p_values = pos, p_time_stamp = time_stamp )
    vel = my_point.get_velocity()
    acc = my_point.get_acceleration()
    my_log.log(Log.C_LOG_TYPE_S, 'pos :', pos, ', vel :', vel, ', acc :', acc)
    time_stamp += time_step

    if visualize: sleep(0.05)



if __name__ == '__main__':
    input('Press ENTER to end the demo...')
