## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - A Synoptic Framework for Standardized Machine Learning Tasks
## -- Package : mlpro.bf.examples
## -- Module  : howto_bf_streams_003_visualize_moving_clouds2d.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2023-03-23  0.0.0     SP       Creation
## -- 2023-03-23  1.0.0     SP       First draft implementation
## -- 2023-04-07  1.0.1     SP       Corrections to include visualize parameter
## -------------------------------------------------------------------------------------------------

"""
Ver. 1.0.2 (2023-03-23)

This module demonstrates how to visualize the moving clouds 2D data stream provided by MLPro.

You will learn:

1) How to access MLPro's native moving clouds 2D data stream.

2) How to iterate the instances of the stream.

3) How to access feature data of the stream.

4) How to visualize the data.

"""

# Import the necessary libraries
from mlpro.bf.streams.streams import *
from mlpro.bf.streams.streams.clouds2d_dynamic import StreamMLProDynamicClouds2D
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mlpro.bf.various import Log



# 0 Prepare Demo/Unit test mode
if __name__ == '__main__':
    logging     = Log.C_LOG_ALL
    visualize   = True
else:
    logging     = Log.C_LOG_NOTHING
    visualize   = False


if visualize:
    # Initialise the stream object using the StreamMLProDynamicClouds3D class
    # and creates an iterator my_iter for it.
    # The argument pattern can be random, random chain, static and merge.
    stream = StreamMLProDynamicClouds2D(p_pattern = 'random', p_variance=7.0 p_logging=logging)
    my_iter = iter(stream)

    # Create three empty lists to store x and y coordinates of the points.
    x = []
    y = []

    # Create the figure and axes objects
    fig, ax = plt.subplots()

    # Iterate over the instances of the stream and store the coordinate values in the empty lists.
    for i, curr_instance in enumerate(my_iter):
        curr_data = curr_instance.get_feature_data().get_values()
        x.append(list(curr_data)[0])
        y.append(list(curr_data)[1])


    # Function that draws each frame of the animation
    def animate(i):
        ax.clear()
        ax.scatter(x[:(i+1)], y[:(i+1)], s=1)
        ax.set_xlim([-110,110])
        ax.set_ylim([-110,110])
        ax.set_xlabel('x')
        ax.set_ylabel('y')


    # Run the animation
    ani = FuncAnimation(fig, animate, frames=999, interval=100, repeat=False)

    plt.show()

