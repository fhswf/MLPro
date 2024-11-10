## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - The integrative middleware framework for standardized machine learning
## -- Package : mlpro.bf.streams.tasks.windows
## -- Module  : ringbuffer.py
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
## -- 2024-05-22  1.2.0     DA       Refactoring, splitting, and renaming to RingBuffer
## -- 2024-05-23  1.2.1     DA       Bugfixes on plotting
## -- 2024-10-31  1.2.2     DA       Bugfix in RingBuffer.get_boundaries()
## -------------------------------------------------------------------------------------------------

"""
Ver. 1.2.2 (2024-10-31)

This module provides pool of window objects further used in the context of online adaptivity.
"""


from matplotlib.axes import Axes
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.patches import Rectangle
import numpy as np
from mlpro.bf.streams.basics import *
from mlpro.bf.events import *
from mlpro.bf.streams.tasks.windows.basics import Window




## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class RingBuffer (Window):
    """
    This class implements a ring buffer.

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

    C_NAME                  = 'Ring Buffer'

    C_PLOT_STANDALONE       = False

    C_PLOT_IN_WINDOW        = 'In Window'
    C_PLOT_OUTSIDE_WINDOW   = 'Out Window'

    C_EVENT_BUFFER_FULL     = 'BUFFER_FULL'     # raised the first time the buffer runs full
    C_EVENT_DATA_REMOVED    = 'DATA_REMOVED'    # raised whenever data were removed from the buffer

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

        super().__init__( p_buffer_size = p_buffer_size,
                          p_delay = p_delay,
                          p_name = p_name,
                          p_range_max = p_range_max,
                          p_duplicate_data = p_duplicate_data,
                          p_visualize = p_visualize,
                          p_logging = p_logging,
                          p_kwargs = p_kwargs )

        self._statistics_enabled        = p_enable_statistics or p_visualize
        self._numeric_buffer:np.ndarray = None
        self._numeric_features          = []
        self._raise_event_data_removed  = False
        self._buffer_full               = False


## -------------------------------------------------------------------------------------------------
    def _run(self, p_inst : InstDict ):
        """
        Method to run the window including adding and deleting of elements

        Parameters
        ----------
        p_inst : InstDict
            Instances to be processed.
        """

        # 0 Intro
        inst = p_inst.copy()
        p_inst.clear()


        # 1 Main processing loop
        for inst_id, (inst_type, inst) in sorted(inst.items()):

            if inst_type != InstTypeNew: 
                # Obsolete instances need to be removed from the buffer (not yet implemented)
                self.log(self.C_LOG_TYPE_W, 'Handling of obsolete data not yet implemented')
                continue


            # 1.1 A new instance is to be buffered
            feature_value  = inst.get_feature_data()


            # 1.2 Checking the numeric dimensions/features in Stream
            if self._numeric_buffer is None and self._statistics_enabled:
                for j in feature_value.get_dim_ids():
                    if feature_value.get_related_set().get_dim(j).get_base_set() in [Dimension.C_BASE_SET_N,
                                                                                     Dimension.C_BASE_SET_R,
                                                                                     Dimension.C_BASE_SET_Z]:
                        self._numeric_features.append(j)

                self._numeric_buffer = np.zeros((self.buffer_size, len(self._numeric_features)))


            # 1.3 Internal ring buffer already filled?
            if len(self._buffer) == self.buffer_size:

                # The oldest instance is extracted from the buffer and forwarded
                inst_del = self._buffer[self._buffer_pos]
                p_inst[inst_del.id] = ( InstTypeDel, inst_del )
                self._raise_event_data_removed = True

                p_inst[inst.id] = ( InstTypeNew, inst )

            elif not self._delay:
                p_inst[inst.id] = ( InstTypeNew, inst )


            # 1.4 New instance is buffered
            self._buffer[self._buffer_pos] = inst
               

            # 1.5 Update of internal statistics
            if self._statistics_enabled:
                self._numeric_buffer[self._buffer_pos] = [feature_value.get_value(k) for k in self._numeric_features]


            # 1.6 Increment of buffer position
            self._buffer_pos = (self._buffer_pos + 1) % self.buffer_size


            # 1.7 Raise events at the end of instance processing
            if ( not self._buffer_full ) and ( len(self._buffer) == self.buffer_size ):
                self._buffer_full = True

                if self._delay:
                    for i in range(self.buffer_size):
                        inst_fwd = self._buffer[i]
                        p_inst[inst_fwd.id] = ( InstTypeNew, inst_fwd )

                self._raise_event( p_event_id = self.C_EVENT_BUFFER_FULL, 
                                   p_event_object = Event( p_raising_object=self, 
                                                           p_related_set=feature_value.get_related_set() ) )


            if self._raise_event_data_removed:
                self._raise_event( p_event_id = self.C_EVENT_DATA_REMOVED, 
                                   p_event_object = Event( p_raising_object=self, 
                                                           p_related_set=feature_value.get_related_set() ) )


## -------------------------------------------------------------------------------------------------
    def get_boundaries(self):
        """
        Method to get the current boundaries of the Window

        Returns
        -------
        boundaries:np.ndarray
            Current window boundaries in the form of a Numpy array.
        """

        if not self._buffer_full:
            boundaries = np.stack( ( [ np.min(self._numeric_buffer[0:self._buffer_pos], axis=0),
                                       np.max(self._numeric_buffer[0:self._buffer_pos], axis=0) ] ), axis=1)
        else:
            boundaries = np.stack( ( [ np.min(self._numeric_buffer, axis=0),
                                       np.max(self._numeric_buffer, axis=0) ] ), axis=1)
            
        return boundaries


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

        Plottable._init_plot_2d(self, p_figure=p_figure, p_settings=p_settings)

        self._patch_windows: dict = None
        self._window_patch2D = Rectangle((0, 0),0,0)
        p_settings.axes.grid(True)


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

        Plottable._init_plot_3d(self, p_figure=p_figure, p_settings=p_settings)

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

        Plottable._init_plot_nd(self, p_figure=p_figure, p_settings=p_settings)

        self._patch_windows = None


## -------------------------------------------------------------------------------------------------
    def _update_plot_2d(self, p_settings:PlotSettings, p_inst:InstDict, **p_kwargs):
        """
        Default 2-dimensional plotting implementation for window tasks. See class mlpro.bf.plot.Plottable
        for more details.

        Parameters
        ----------
        p_settings : PlotSettings
            Object with further plot settings.
        p_inst : InstDict
            Stream instances to be plotted.
        p_kwargs : dict
            Further optional plot parameters.
        """

        # 1 No visualization until first data is buffered
        if len(self._buffer) == 0: return


        # 2 Initialization of the rectangle
        if self._patch_windows is None:
            self._patch_windows = {}
            self._patch_windows['2D'] = Rectangle((0, 0),0,0, ec= 'red', facecolor='none', zorder = -999)
            self._plot_settings.axes.add_patch(self._patch_windows['2D'])
            self._patch_windows['2D'].set_visible(True)


        # 3 Update of the rectangle
        boundaries = self.get_boundaries()
        x = boundaries[0][0]
        y = boundaries[1][0]
        w = boundaries[0][1] - boundaries[0][0]
        h = boundaries[1][1] - boundaries[1][0]
        self._patch_windows['2D'].set_bounds(x,y,w,h)

        self._patch_windows['2D'].set_visible(True)


## -------------------------------------------------------------------------------------------------
    def _update_plot_3d(self, p_settings:PlotSettings, p_inst:InstDict, **p_kwargs):
        """
        Default 3-dimensional plotting implementation for window tasks. See class mlpro.bf.plot.Plottable
        for more details.

        Parameters
        ----------
        p_settings : PlotSettings
            Object with further plot settings.
        p_inst : InstDict
            Stream instances to be plotted.
        p_kwargs : dict
            Further optional plot parameters.
        """

        # 1 No visualization until first data is buffered
        if len(self._buffer) == 0: return


        # 2 Initialization of the cuboid
        if self._patch_windows is None:
            self._patch_windows = {}
            self._patch_windows['3D'] = Poly3DCollection(verts= [], edgecolors='red', facecolors='red', alpha = 0)
            self._plot_settings.axes.add_collection(self._patch_windows['3D'])


        # 3 Update of the cuboid
        b = self.get_boundaries()

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

        self._patch_windows['3D'].set_verts(verts)


## -------------------------------------------------------------------------------------------------
    def _update_plot_nd(self, p_settings:PlotSettings, p_inst:InstDict, **p_kwargs):
        """
        The n-dimensional representation of the ring buffer visualizes the removal of obsolete data 
        from the buffer by hiding it behind a semi-transparent rectangle. The visualization starts 
        when the buffer is completely filled and data is removed.

        Parameters
        ----------
        p_settings : PlotSettings
            Object with further plot settings.
        p_inst : InstDict
            Stream instances to be plotted.
        p_kwargs : dict
            Further optional plot parameters.
        """

        # 1 No visualization until the buffer has been filledd
        if ( not self._buffer_full ) or ( len(p_inst) == 0 ): return


        # 2 Check if the rectangle patches are already created
        if self._patch_windows is None:
            self._patch_windows = {}

            bg = p_settings.axes.get_facecolor()
            ec = p_settings.axes.patch.get_edgecolor()
            obs_window = Rectangle((0,0), 0,0, facecolor = bg, edgecolor=ec, lw = 1, zorder=9999, alpha = 0.75 )
            p_settings.axes.add_patch(obs_window)
            self._patch_windows['nD'] = obs_window


        # 3 Add the hiding plot around obsolete data
        inst_oldest = self._buffer[self._buffer_pos]
        x = p_settings.axes.get_xlim()[0]
        y = p_settings.axes.get_ylim()[0]
        w = inst_oldest.tstamp - x
        h = p_settings.axes.get_ylim()[1] - y
        self._patch_windows['nD'].set_bounds(x, y, w, h)
        self._patch_windows['nD'].set_visible(True)
