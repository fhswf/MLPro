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
## -- 2022-12-16  1.0.2     LSB      Delay in delivering the buffered data
## -- 2022-12-16  1.0.3     DA       Refactoring after changes on bf.streams
## -- 2022-12-18  1.0.4     LSB      Bug Fixes
## -- 2022-12-18  1.1.0     LSB      New plot updates -
##                                   - single rectangle
##                                   - transparent patch on obsolete data
## -- 2022-12-19  1.1.1     DA       New parameter p_duplicate_data
## -- 2022-12-28  1.1.2     DA       Refactoring of plot settings
## -- 2022-12-29  1.1.3     DA       Removed method Window.init_plot()
## -- 2022-12-31  1.1.4     LSB      Refactoring
## -- 2023-02-02  1.1.5     DA       Methods Window._init_plot_*: removed figure creation
## -------------------------------------------------------------------------------------------------

"""
Ver. 1.1.5 (2023-02-02)

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
        p_duplicate_data : bool
            If True, instances will be duplicated before processing. Default = False.
        p_ada:bool, optional
            Adaptivity property of object. Default is True.
        p_logging      -Optional
            Log level for the object. Default is log everything.
    """

    C_NAME                  = 'Window'

    C_PLOT_STANDALONE       = False

    C_PLOT_IN_WINDOW        = 'In Window'
    C_PLOT_OUTSIDE_WINDOW   = 'Out Window'

    C_EVENT_BUFFER_FULL     = 'BUFFER_FULL'
    C_EVENT_DATA_REMOVED    = 'DATA_REMOVED'

## -------------------------------------------------------------------------------------------------
    def __init__(self,
                 p_buffer_size:int,
                 p_delay:bool = False,
                 p_enable_statistics:bool = False,
                 p_name:str   = None,
                 p_range_max  = StreamTask.C_RANGE_THREAD,
                 p_duplicate_data : bool = False,
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
    def _run(self, p_inst_new:list, p_inst_del:list ):
        """
        Method to run the window including adding and deleting of elements

        Parameters
        ----------
        p_inst_new : list
            Instance/s to be added to the window
        p_inst_del : list
            Instance/s to be deleted from the window
        """

        # 1 Checking if there are new instances
        if p_inst_new:
            new_instances = p_inst_new.copy()
            num_inst = len(new_instances)


            for i in range(num_inst):
                inst = new_instances[i]


                # Compatibility with Instance/State
                if isinstance(inst, Instance):
                    feature_value = inst.get_feature_data()
                else:
                    feature_value = inst


                # Checking the numeric dimensions/features in Stream
                if self._numeric_buffer is None and self._statistics_enabled:
                    for j in feature_value.get_dim_ids():
                        if feature_value.get_related_set().get_dim(j).get_base_set() in [Dimension.C_BASE_SET_N,
                                                                                         Dimension.C_BASE_SET_R,
                                                                                         Dimension.C_BASE_SET_Z]:
                            self._numeric_features.append(j)

                    self._numeric_buffer = np.zeros((self.buffer_size, len(self._numeric_features)))


                # Increment in buffer position
                self._buffer_pos = (self._buffer_pos + 1) % self.buffer_size


                if len(self._buffer) == self.buffer_size:
                    # if the buffer is already full,obsolete data is going to be deleted
                    # raises an event, stores the new instances and skips the iteration
                    self._raise_event(self.C_EVENT_DATA_REMOVED, Event(p_raising_object=self,
                                                                       p_related_set=feature_value.get_related_set()))
                    p_inst_del.append(self._buffer[self._buffer_pos])
                    self._buffer[self._buffer_pos] = inst
                    if self._statistics_enabled:
                        self._numeric_buffer[self._buffer_pos] = [feature_value.get_value(k) for k in
                                                                  self._numeric_features]
                    continue


                # adds element to the buffer in this code block only if the buffer is not already full
                self._buffer[self._buffer_pos] = inst
                if self._statistics_enabled:
                    self._numeric_buffer[self._buffer_pos] = [feature_value.get_value(k) for k in
                                                              self._numeric_features]


                # if the buffer is full after adding an element, raises event
                if len(self._buffer) == self.buffer_size:
                    if self._delay:
                        p_inst_new.clear()
                        for instance in self._buffer.values():
                            p_inst_new.append(instance)
                    self._raise_event(self.C_EVENT_BUFFER_FULL, Event(self))


            # If delay is true, clear the set p_inst_new for any following tasks
            if self._delay:
                if len(self._buffer) < self.buffer_size:
                    p_inst_new.clear()


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
        if self._delay:
            if len(self._buffer) < self.buffer_size: return
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

        if p_settings:
            self._plot_settings = p_settings

        if not p_settings.axes:
            self.axes = Axes(p_figure, [0.05,0.05,0.9,0.9])

        else:
            self.axes = p_settings.axes
        self._patch_windows: dict = None
        self._window_patch2D = Rectangle((0, 0),0,0)


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


        if p_settings:
            self._plot_settings = p_settings

        if not p_settings.axes:
            self.axes = p_figure.add_subplot(projection = '3d')
        else:
            self.axes = p_settings.axes

        self._patch_windows: dict = None
        self._window_patch3D = Poly3DCollection([])


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

        if p_settings:
            self._plot_settings = p_settings

        if not p_settings.axes:
            self.axes = p_figure.add_subplot()
            p_settings.axes = self.axes
            p_settings.axes.set_xlabel(self.C_PLOT_ND_XLABEL_INST)
            p_settings.axes.set_ylabel(self.C_PLOT_ND_YLABEL)
            p_settings.axes.grid(visible=True)

        else:
            self.axes = p_settings.axes

        self._patch_windows = None


## -------------------------------------------------------------------------------------------------
    def _update_plot_2d(self, p_settings:PlotSettings, p_inst_new:list, p_inst_del:list, **p_kwargs):
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
        self.axes.grid(True)
        if self._patch_windows is None:
            self._patch_windows = {}
            self._patch_windows['2D'] = Rectangle((0, 0),0,0, ec= 'red', facecolor='none', zorder = -999)
            self._plot_settings.axes.add_patch(self._patch_windows['2D'])
            self._patch_windows['2D'].set_visible(True)

        boundaries = self.get_boundaries()
        x = boundaries[0][0]
        y = boundaries[1][0]
        w = boundaries[0][1] - boundaries[0][0]
        h = boundaries[1][1] - boundaries[1][0]
        self._patch_windows['2D'].set_bounds(x,y,w,h)

        self._patch_windows['2D'].set_visible(True)


## -------------------------------------------------------------------------------------------------
    def _update_plot_3d(self, p_settings:PlotSettings, p_inst_new:list, p_inst_del:list, **p_kwargs):
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

        if self._patch_windows is None:
            self._patch_windows = {}
            self._patch_windows['3D'] = Poly3DCollection(verts= [], edgecolors='red', facecolors='red', alpha = 0)
            self._plot_settings.axes.add_collection(self._patch_windows['3D'])

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
        self._patch_windows['3D'].set_verts(verts)


## -------------------------------------------------------------------------------------------------
    def _update_plot_nd(self, p_settings:PlotSettings, p_inst_new:list, p_inst_del:list, **p_kwargs):
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
        if len(p_inst_new) == 0 : return

        # 2. Check if the rectangle patches are already created
        if self._patch_windows is None:
            self._patch_windows = {}

            bg = self.axes.get_facecolor()
            ec = self.axes.patch.get_edgecolor()
            obs_window = Rectangle((0,0), 0,0, facecolor = bg, edgecolor=ec, lw = 1, zorder=9999, alpha = 0.75 )
            self._plot_settings.axes.add_patch(obs_window)
            self._patch_windows['nD'] = obs_window


        # 3. Add the hiding plot around obsolete data
        x1 = self._plot_num_inst-self.buffer_size+1
        y1 = self.axes.get_ylim()[0]
        w1 = -(x1 - self.axes.get_xlim()[0])
        h1 = self.axes.get_ylim()[1] - y1
        self._patch_windows['nD'].set_bounds(x1, y1, w1, h1)
        self._patch_windows['nD'].set_visible(True)
