## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - A Synoptic Framework for Standardized Machine Learning Tasks
## -- Package : mlpro.bf.streams.tasks.windows
## -- Module  : windows.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2022-10-16  0.0.0     LSB      Creation
## -- 2022-11-04  0.1.0     LSB      Removing class WindowR
## -- 2022-11-24  0.2.0     LSB      Implementations and release of nd plotting
## -- 2022-11-26  0.3.0     LSB      Implementations and release of 3-d plotting
## -- 2022-12-08  0.4.0     DA       Refactoring after changes on bf.streams
## -- 2022-12-08  1.0.0     LSB      Release
## -- 2022-12-08  1.0.1     LSB      Compatilbility for both Instance and Element object
## -------------------------------------------------------------------------------------------------

"""
Ver. 1.0.1 (2022-12-08)
This module provides pool of window objects further used in the context of online adaptivity.
"""


from matplotlib.axes import Axes
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.collections import PolyCollection
from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt
import numpy as np
from mlpro.bf.streams.models import *
from mlpro.bf.events import *
from typing import Union, List, Iterable
import matplotlib.colors as colors





## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class Window (StreamTask):
    """
    This is the base class for window implementations

    Parameters
    ----------
        p_buffer_size:int
            the size/length of the buffer/window.
        p_delay:bool, optional
            Set to true if full buffer is desired before passing the window data to next step. Default is false.
        p_name:str, optional
            Name of the Window. Default is None.
        p_range_max     -Optional
            Maximum range of task parallelism for window task. Default is set to multithread.
        p_ada:bool, optional
            Adaptivity property of object. Default is True.
        p_logging      -Optional
            Log level for the object. Default is log everything.
    """

    C_NAME                  = 'Window'

    C_PLOT_STANDALONE       = False

    C_EVENT_BUFFER_FULL     = 'BUFFER_FULL'
    C_EVENT_DATA_REMOVED    = 'DATA_REMOVED'

## -------------------------------------------------------------------------------------------------
    def __init__(self,
                 p_buffer_size:int,
                 p_delay:bool = False,
                 p_enable_statistics:bool = False,
                 p_name:str   = None,
                 p_range_max  = StreamTask.C_RANGE_THREAD,
                 p_duplicate_data:bool = False,
                 p_visualize:bool = False,
                 p_logging    = Log.C_LOG_ALL,
                 **p_kwargs):

        self._kwargs     = p_kwargs.copy()
        self.buffer_size = p_buffer_size
        self._delay      = p_delay
        self._name       = p_name
        self._range_max  = p_range_max
        self.switch_logging(p_logging = p_logging)

        super().__init__(p_name      = p_name,
                         p_range_max = p_range_max,
                         p_duplicate_data = p_duplicate_data,
                         p_visualize = p_visualize,
                         p_logging   = p_logging)

        self._buffer = {}
        self._buffer_pos = 0
        self._statistics_enabled = p_enable_statistics or p_visualize
        self._numeric_buffer:np.ndarray = None
        self._numeric_features = []


## -------------------------------------------------------------------------------------------------
    def _run(self, p_inst_new:set, p_inst_del:set ):
        """
        Method to run the window including adding and deleting of elements

        Parameters
        ----------
        p_inst_new : set
            Instance/s to be added to the window
        p_inst_del : set
            Instance/s to be deleted from the window
        """

        if p_inst_new:
            for i in p_inst_new:
                if isinstance(i, Instance):
                    i = i.get_feature_data()
                if self._numeric_buffer is None and self._statistics_enabled:
                    for j in i.get_dim_ids():
                        if i.get_related_set().get_dim(j).get_base_set() in [Dimension.C_BASE_SET_N,
                                                                             Dimension.C_BASE_SET_R,
                                                                             Dimension.C_BASE_SET_Z]:
                            self._numeric_features.append(j)

                    self._numeric_buffer = np.zeros((self.buffer_size, len(self._numeric_features)))

                self._buffer_pos = (self._buffer_pos + 1) % self.buffer_size

                if len(self._buffer) == self.buffer_size:
                    # Checks if the buffer is already full, implying that obsolete data is going to be deleted and
                    # raises an event, and stores the new instances and continues the loop

                    self._raise_event(self.C_EVENT_DATA_REMOVED, Event(p_raising_object=self,
                                                                       p_related_set=i.get_feature_data().
                                                                           get_related_set()))
                    p_inst_del.add(self._buffer[self._buffer_pos])
                    self._buffer[self._buffer_pos] = i
                    if self._statistics_enabled:
                        self._numeric_buffer[self._buffer_pos] = [i.get_value(k) for k in self._numeric_features]
                    continue

                # adds element to the buffer
                self._buffer[self._buffer_pos] = i
                if self._statistics_enabled:
                    self._numeric_buffer[self._buffer_pos] = [i.get_value(k) for k in self._numeric_features]

                # if the buffer is full after adding an element, raises event
                if len(self._buffer) == self.buffer_size:
                    self._raise_event(self.C_EVENT_BUFFER_FULL, Event(self))


## -------------------------------------------------------------------------------------------------
    def get_buffered_data(self):
        """
        Method to fetch the date from the window buffer

        Returns
        -------
            buffer:dict
                the buffered data in the form of dictionary
            buffer_pos:int
                the latest buffer position
        """

        return self._buffer, self._buffer_pos


## -------------------------------------------------------------------------------------------------
    def get_boundaries(self):
        """
        Method to get the current boundaries of the Window

        Returns
        -------
            boundaries:np.ndarray
                Returns the current window boundaries in the form of a Numpy array.
        """
        boundaries = np.stack(([np.min(self._numeric_buffer, axis=0),
                      np.max(self._numeric_buffer, axis=0)]), axis=1)
        return boundaries


## -------------------------------------------------------------------------------------------------
    def get_mean(self):
        """
        Method to get the mean of the data in the Window.

        Returns
        -------
            mean:np.ndarray
                Returns the mean of the current data in the window in the form of a Numpy array.
        """

        return np.mean(self._buffer.values(), axis=0, dtype=np.float64)


## -------------------------------------------------------------------------------------------------
    def get_variance(self):
        """
        Method to get the variance of the data in the Window.

        Returns
        -------
            variance:np.ndarray
                Returns the variance of the current data in the window as a numpy array.
        """

        return np.variance(self._buffer.values(), axis=0, dtype=np.float64)


## -------------------------------------------------------------------------------------------------
    def get_std_deviation(self):
        """
        Method to get the standard deviation of the data in the window.

        Returns
        -------
            std:np.ndarray
                Returns the standard deviation of the data in the window as a numpy array.
        """

        return np.std(self._buffer.values(), axis=0, dtype=np.float64)


## -------------------------------------------------------------------------------------------------
    def init_plot(self,
                      p_figure: Figure = None,
                      p_plot_settings: list = [],
                      p_plot_depth: int = 0,
                      p_detail_level: int = 0,
                      p_step_rate: int = 0,
                      **p_kwargs):
        """
        Method to initialize the plot for Window task.

        Parameters
        ----------
        p_figure:Figure
            Figure to host the plot.
        p_plot_settings:PlotSettings
            Specific plot settings
        p_plot_depth:
            Depth of plotting style
        p_detail_level:
            Level of details for the plot style
        p_step_rate:
            Rate of update for updating the plot
        p_kwargs:
            Additional key-worded arguments
        """

        self._plot_num_inst = 0

        Task.init_plot(self,
                p_figure=p_figure,
                p_plot_settings=p_plot_settings,
                p_plot_depth=p_plot_depth,
                p_detail_level=p_detail_level,
                p_step_rate=p_step_rate,
                **p_kwargs)


## -------------------------------------------------------------------------------------------------
    def _init_plot_2d(self, p_figure: Figure, p_settings: PlotSettings):
        """
        Custom method to initialize a 2D plot for the window object

        Parameters
        ----------
        p_figure: Figure
            The figure object that hosts the plot

        p_settings: list of PlotSettings objects.
            Additional settings for the plot

        """

        if p_figure is None:
            p_figure = plt.figure()

        if not p_settings.axes:
            self.axes = Axes(p_figure, [0.05,0.05,0.9,0.9])
        else:
            self.axes = p_settings.axes

        self.window_patch2D = Rectangle((0,0),0,0)

        pass


## -------------------------------------------------------------------------------------------------
    def _init_plot_3d(self, p_figure: Figure, p_settings: PlotSettings):
        """
        Custom method to initialize a 3D plot for window object

        Parameters
        ----------
        p_figure: matplotlib.figure.Figure
            The figure object to host the plot.

        p_settings: PlotSettings
            Additional Settings for the plot
        """
        if p_figure is None:
            p_figure = plt.figure()

        if not p_settings.axes:
            self.axes = Axes3D(p_figure, (0.05,0.05,0.9,0.9))
        else:
            self.axes = p_settings.axes

        self.window_patch3D = Poly3DCollection([])

        # ...


## -------------------------------------------------------------------------------------------------
    def _init_plot_nd(self, p_figure: Figure, p_settings: PlotSettings):
        """
        Custom method to initialize plot for Window tasks for N-dimensional plotting.

        Parameters
        ----------
        p_figure:Figure
            Figure to host the plot
        p_settings: PlotSettings
            PlotSettings objects with specific settings for the plot

        """
        if p_figure is None:
            p_figure = plt.figure()

        if not p_settings.axes:
            self.axes = Axes(p_figure, [0.05,0.05,0.9,0.9])
            p_settings.axes.set_xlabel(self.C_PLOT_ND_XLABEL_INST)
            p_settings.axes.set_ylabel(self.C_PLOT_ND_YLABEL)
            p_settings.axes.grid(visible=True)

        else:
            self.axes = p_settings.axes

        self._plot_nd_plots = None
        self._plot_windows = []


## -------------------------------------------------------------------------------------------------
    def _update_plot_2d(self, p_settings:PlotSettings, p_inst_new:set, p_inst_del:set, **p_kwargs):
        """

        Parameters
        ----------
        p_settings
        p_inst_new
        p_inst_del
        p_kwargs

        Returns
        -------

        """
        boundaries = self.get_boundaries()
        x = boundaries[0][0]
        y = boundaries[1][0]
        w = boundaries[0][1] - boundaries[0][0]
        h = boundaries[1][1] - boundaries[1][0]
        self.window_patch2D.set_bounds(x,y,w,h)


## -------------------------------------------------------------------------------------------------
    def _update_plot_3d(self, p_settings:PlotSettings, p_inst_new:set, p_inst_del:set, **p_kwargs):
        """

        Default 3-dimensional plotting implementation for window tasks. See class mlpro.bf.plot.Plottable
        for more details.

        Parameters
        ----------
        p_settings : PlotSettings
            Object with further plot settings.
        p_inst_new : list
            List of new stream instances to be plotted.
        p_inst_del : list
            List of obsolete stream instances to be removed.
        p_kwargs : dict
            Further optional plot parameters.

        """
        # 1. Returns if no new instances passed
        if p_inst_new is None: return
        b = self.get_boundaries()

        # 2. Logic for vertices of the cuboid
        verts = np.asarray([[[b[0][0], b[1][0], b[2][1]],
                             [b[0][1], b[1][0], b[2][1]],
                             [b[0][1], b[1][0], b[2][0]],
                             [b[0][0], b[1][0], b[2][0]]],

                            [[b[0][0], b[1][0], b[2][1]],
                             [b[0][1], b[1][0], b[2][1]],
                             [b[0][1], b[1][1], b[2][1]],
                             [b[0][0], b[1][1], b[2][1]]],

                            [[b[0][0], b[1][0], b[2][1]],
                             [b[0][0], b[1][1], b[2][1]],
                             [b[0][0], b[1][1], b[2][0]],
                             [b[0][0], b[1][0], b[2][0]]],

                            [[b[0][1], b[1][0], b[2][1]],
                             [b[0][1], b[1][1], b[2][1]],
                             [b[0][1], b[1][1], b[2][0]],
                             [b[0][1], b[1][0], b[2][0]]],

                            [[b[0][0], b[1][1], b[2][1]],
                             [b[0][1], b[1][1], b[2][1]],
                             [b[0][1], b[1][1], b[2][0]],
                             [b[0][0], b[1][1], b[2][0]]],

                            [[b[0][0], b[1][0], b[2][0]],
                             [b[0][1], b[1][0], b[2][0]],
                             [b[0][1], b[1][1], b[2][0]],
                             [b[0][0], b[1][1], b[2][0]]]])

        # 3. Setting the vertices for the cuboid
        self.window_patch3D.set_verts(verts)


## -------------------------------------------------------------------------------------------------
    def _update_plot_nd(self, p_settings:PlotSettings, p_inst_new:set, p_inst_del:set, **p_kwargs):
        """
        Default N-dimensional plotting implementation for window tasks. See class mlpro.bf.plot.Plottable
        for more details.

        Parameters
        ----------
        p_settings : PlotSettings
            Object with further plot settings.
        p_inst_new : list
            List of new stream instances to be plotted.
        p_inst_del : list
            List of obsolete stream instances to be removed.
        p_kwargs : dict
            Further optional plot parameters.
        """

        # 1. CHeck if there is a new instance to be plotted
        if p_inst_new is None: return

        # 2. Check if the rectangle patches are already created
        inst_new = list(p_inst_new)

        if self._plot_nd_plots is None:
            self._plot_nd_plots = {}

            color_names = list(colors.cnames.values())

        # If not create new patch objects and add them to the attribute
            feature_space = inst_new[0].get_feature_data().get_related_set()
            for i,feature in enumerate(feature_space.get_dims()):
                if feature.get_base_set() in [Dimension.C_BASE_SET_R, Dimension.C_BASE_SET_N, Dimension.C_BASE_SET_Z]:
                    feature_window = Rectangle((0,0), 0,0, facecolor = color_names[i], edgecolor=color_names[i],
                        lw = 1.5, alpha = 0.5)
                    self.axes.add_patch(feature_window)
                    self._plot_nd_plots[feature.get_id()] = feature_window

        # 2. Getting the boundaries
        boundaries = self.get_boundaries()

        # 3. Adding rectangular patches to the plots
        for i,feature in enumerate(inst_new[0].get_feature_data().get_related_set().get_dims()):
            if feature.get_base_set() in [Dimension.C_BASE_SET_R, Dimension.C_BASE_SET_N, Dimension.C_BASE_SET_Z]:
                x = self._plot_num_inst-self.buffer_size+1
                y = boundaries[i][0]
                w = self.buffer_size
                h = boundaries[i][1] - boundaries[i][0]
                self._plot_nd_plots[feature.get_id()].set_bounds(x,y,w,h)
                self._plot_nd_plots[feature.get_id()].set_visible(True)

