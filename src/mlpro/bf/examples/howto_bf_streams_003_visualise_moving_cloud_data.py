## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - A Synoptic Framework for Standardized Machine Learning Tasks
## -- Package : mlpro.bf.examples
## -- Module  : howto_bf_streams_001_accessing_native_data_from_mlpro.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2023-03-23  1.0.0     DA       Creation
## -- 2023-03-23  1.0.1     DA       Corrections
## -------------------------------------------------------------------------------------------------

"""
Ver. 1.0.0 (2023-03-23)

This module demonstrates the use of native generic data streams provided by MLPro. To this regard,
all data streams of the related provider class will be determined and iterated. 

You will learn:

1) How to access MLPro's native data streams.

2) How to iterate the instances of a native stream.

3) How to access feature data of a native stream.

"""


from mlpro.bf.streams.streams import *
from mlpro.bf.streams.streams.clouds2d_dynamic import StreamMLProDynamicClouds2D
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation




stream = StreamMLProDynamicClouds2D(pattern = 'random chain')
my_iter = iter(stream)

dataset = []
x = []
y = []

# create the figure and axes objects
fig, ax = plt.subplots()

for i, curr_instance in enumerate(my_iter):
    curr_data = curr_instance.get_feature_data().get_values()
    #dataset.append(list(curr_data))
    x.append(list(curr_data)[0])
    y.append(list(curr_data)[1])


# function that draws each frame of the animation
def animate(i):
    ax.clear()
    ax.scatter(x[:(i+1)], y[:(i+1)])
    ax.set_xlim([-20,20])
    ax.set_ylim([-20,20])
    ax.set_xlabel('x')
    ax.set_ylabel('y')

# run the animation
ani = FuncAnimation(fig, animate, frames=999, interval=100, repeat=False)

plt.show()

