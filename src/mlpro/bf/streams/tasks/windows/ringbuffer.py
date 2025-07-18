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
## -- 2024-12-11  1.2.3     DA       Pseudo classes if matplotlib is not installed
## -- 2025-04-11  1.2.4     DA       - Code review/cleanup
## --                                - Method RingBuffer._update_plot_nd(): support of time stamps
## -- 2025-05-06  1.2.5     DA       Method RingBuffer._run(): update tstamp of outdated instances
## -- 2025-06-05  2.0.0     DA       Refactoring of class RingBuffer
## --                                - method get_boundaries(): alignment to new signature
## --                                - removal of event propagation
## --                                - optimization of methods _update_plot_2d(), _update_plot_3d()
## -- 2025-06-06  2.1.0     DA       Refactoring: p_inst -> p_instances
## -- 2025-06-08  2.2.0     DA       Refactoring of methods RingBuffer._update_plot*: new return param
## -- 2025-06-24  2.3.0     DA       Optimized method RingBuffer._update_plot_nd(): 
## -- 2025-06-27  2.4.0     DA       Method RingBuffer._update_plot_nd(): set box color to lightgrey    
## -------------------------------------------------------------------------------------------------

"""
Ver. 2.4.0 (2025-06-27)

This module provides a sliding window with an internal ring buffer.
"""

from typing import Union

import numpy as np

try:
    from matplotlib.figure import Figure
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    from matplotlib.patches import Rectangle
    import matplotlib.dates as mdates
except:
    class Poly3DCollection : pass
    class Rectangle : pass

from mlpro.bf.various import Log
from mlpro.bf.plot import PlotSettings, Plottable
from mlpro.bf.math import Dimension
from mlpro.bf.math.statistics import Boundaries, BoundarySide
from mlpro.bf.streams import InstDict, InstTypeNew, InstTypeDel, StreamTask
from mlpro.bf.streams.tasks.windows.basics import Window



# Export list for public API
__all__ = [ 'RingBuffer' ]




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

    C_NAME      = 'Ring Buffer'

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
        self._buffer_full               = False
        self._boundaries : Boundaries   = None


## -------------------------------------------------------------------------------------------------
    def _run(self, p_instances : InstDict ):
        """
        Method to run the window including adding and deleting of elements

        Parameters
        ----------
        p_instances : InstDict
            Instances to be processed.
        """

        # 0 Intro
        instances = p_instances.copy()
        p_instances.clear()


        # 1 Main processing loop
        for inst_id, (inst_type, instance) in sorted(instances.items()):

            if inst_type != InstTypeNew: 
                # Obsolete instances need to be removed from the buffer (not yet implemented)
                self.log(self.C_LOG_TYPE_W, 'Handling of obsolete data not yet implemented')
                continue


            # 1.1 A new instance is to be buffered
            feature_data  = instance.get_feature_data()


            # 1.2 Checking the numeric dimensions/features in Stream
            if self._numeric_buffer is None and self._statistics_enabled:
                for j in feature_data.get_dim_ids():
                    if feature_data.get_related_set().get_dim(j).get_base_set() in [ Dimension.C_BASE_SET_N,
                                                                                     Dimension.C_BASE_SET_R,
                                                                                     Dimension.C_BASE_SET_Z ]:
                        self._numeric_features.append(j)

                self._numeric_buffer = np.full((self.buffer_size, len(self._numeric_features)), np.nan)


            # 1.3 Internal ring buffer already filled?
            if len(self._buffer) == self.buffer_size:

                # The oldest instance is extracted from the buffer and forwarded
                instance_del = self._buffer[self._buffer_pos]
                instance_del.tstamp = self.get_so().tstamp
                p_instances[instance_del.id] = ( InstTypeDel, instance_del )
                p_instances[instance.id] = ( InstTypeNew, instance )

            elif not self._delay:
                p_instances[instance.id] = ( InstTypeNew, instance )


            # 1.4 New instance is buffered
            self._buffer[self._buffer_pos] = instance
               

            # 1.5 Update of internal statistics
            if self._statistics_enabled:
                self._numeric_buffer[self._buffer_pos] = [feature_data.get_value(k) for k in self._numeric_features]


            # 1.6 Increment of buffer position
            self._buffer_pos = (self._buffer_pos + 1) % self.buffer_size


            # 1.7 Raise events at the end of instance processing
            if ( not self._buffer_full ) and ( len(self._buffer) == self.buffer_size ):
                self._buffer_full = True

                if self._delay:
                    for i in range(self.buffer_size):
                        instance_fwd = self._buffer[i]
                        p_instances[instance_fwd.id] = ( InstTypeNew, instance_fwd )


## -------------------------------------------------------------------------------------------------
    def get_boundaries( self, 
                        p_dim : int = None,
                        p_side : BoundarySide = None,
                        p_copy : bool = False ) -> Union[Boundaries, float]:
        """
        Returns the current value boundaries of internally stored data. The result can be reduced
        by the optional parameters p_side, p_dim. If both parameters are specified, the result is
        a float.

        Parameters
        ----------
        p_dim : int = None
            Optionally reduces the result to a particular dimension.
        p_side : BoundarySide = None
            Optionally reduces the result to upper or lower boundaries. See class BoundarySide for
            possible values.
        p_copy : bool = False
            If True, a copy of the boudaries is returned. Otherwise (default), a reference to the
            internal boundary array is returned.

            
        Returns
        -------
        Union[Boundaries, float]
            Returns the current boundaries of the data. The return value depends on the combination 
            of the optional parameters:
            - If neither `p_side` nor `p_dim` is specified: returns the full 2×n array.
            - If only `p_side` is specified: returns a 1D array with values for all dimensions.
            - If only `p_dim` is specified: returns a 1D array with both lower and upper bounds.
            - If both `p_side` and `p_dim` are specified: returns a single float value.
        """

        # 1 Initialize boundary structure if not yet done
        if self._boundaries is None:
            try:
                num_dims = self._numeric_buffer.shape[1]
                self._boundaries = self._create_boundaries(p_num_dim=num_dims)
            except:
                return None
            

        # 2 Selective update and return
        if p_side is None:
            if p_dim is None:
                # Case 1: Full boundaries array for all dimensions and both sides
                np.nanmin(self._numeric_buffer, axis=0, out=self._boundaries[:, BoundarySide.LOWER])
                np.nanmax(self._numeric_buffer, axis=0, out=self._boundaries[:, BoundarySide.UPPER])
                result = self._boundaries

            else:
                # Case 2: Both sides for one specific dimension (as 1D array)
                self._boundaries[p_dim, BoundarySide.LOWER] = np.nanmin(self._numeric_buffer[:, p_dim])
                self._boundaries[p_dim, BoundarySide.UPPER] = np.nanmax(self._numeric_buffer[:, p_dim])
                result = self._boundaries[p_dim, :]

        else:
            if p_dim is None:
                # Case 3: One side for all dimensions (as 1D array)
                func = np.nanmin if p_side == BoundarySide.LOWER else np.nanmax
                func(self._numeric_buffer, axis=0, out=self._boundaries[:, p_side])
                result = self._boundaries[:, p_side]

            else:
                # Case 4: One side for one specific dimension (as scalar)
                func = np.nanmin if p_side == BoundarySide.LOWER else np.nanmax
                self._boundaries[p_dim, p_side] = func(self._numeric_buffer[:, p_dim])
                return self._boundaries[p_dim, p_side]


        # 3 Return result (copy if requested)
        return result.copy() if p_copy else result
    

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
        self._plot_boundaries     = None
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
        self._plot_boundaries     = None
        

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

        self._patch_windows       = None
        self._plot_boundaries     = None


## -------------------------------------------------------------------------------------------------
    def _update_plot_2d(self, p_settings:PlotSettings, p_instances:InstDict, **p_kwargs) -> bool:
        """
        Default 2-dimensional plotting implementation for window tasks. See class mlpro.bf.plot.Plottable
        for more details.

        Parameters
        ----------
        p_settings : PlotSettings
            Object with further plot settings.
        p_instances : InstDict
            Stream instances to be plotted.
        p_kwargs : dict
            Further optional plot parameters.

        Returns
        -------
        bool   
            True, if changes on the plot require a refresh of the figure. False otherwise.          
        """

        # 1 No visualization until first data is buffered
        if len(self._buffer) == 0: return False


        # 2 Check for changes on boundaries
        plot_boundaries = self.get_boundaries().copy()
        if ( self._plot_boundaries is not None ) and np.array_equal( plot_boundaries, self._plot_boundaries ): return
        self._plot_boundaries = plot_boundaries


        # 3 Initialization of the rectangle
        if self._patch_windows is None:
            self._patch_windows = {}
            self._patch_windows['2D'] = Rectangle((0, 0),0,0, ec= 'red', facecolor='none', zorder = -999)
            self._plot_settings.axes.add_patch(self._patch_windows['2D'])
            self._patch_windows['2D'].set_visible(True)


        # 4 Update of the rectangle
        x = plot_boundaries[0,BoundarySide.LOWER]
        y = plot_boundaries[1,BoundarySide.LOWER]
        w = plot_boundaries[0,BoundarySide.UPPER] - plot_boundaries[0,BoundarySide.LOWER]
        h = plot_boundaries[1,BoundarySide.UPPER] - plot_boundaries[1,BoundarySide.LOWER]
        self._patch_windows['2D'].set_bounds(x,y,w,h)

        self._patch_windows['2D'].set_visible(True)

        return True


## -------------------------------------------------------------------------------------------------
    def _update_plot_3d(self, p_settings:PlotSettings, p_instances:InstDict, **p_kwargs) -> bool:
        """
        Default 3-dimensional plotting implementation for window tasks. See class mlpro.bf.plot.Plottable
        for more details.

        Parameters
        ----------
        p_settings : PlotSettings
            Object with further plot settings.
        p_instances : InstDict
            Stream instances to be plotted.
        p_kwargs : dict
            Further optional plot parameters.

        Returns
        -------
        bool   
            True, if changes on the plot require a refresh of the figure. False otherwise.          
        """

        # 1 No visualization until first data is buffered
        if len(self._buffer) == 0: return False


        # 2 Check for changes on boundaries
        plot_boundaries = self.get_boundaries().copy()
        if ( self._plot_boundaries is not None ) and np.array_equal( plot_boundaries, self._plot_boundaries ): return
        self._plot_boundaries = plot_boundaries
            
            
        # 3 Initialization of the cuboid
        if self._patch_windows is None:
            self._patch_windows = {}
            self._patch_windows['3D'] = Poly3DCollection(verts= [], edgecolors='red', facecolors='red', alpha = 0)
            self._plot_settings.axes.add_collection(self._patch_windows['3D'])


        # 4 Update of the cuboid
        b     = plot_boundaries
        verts = np.asarray([[[b[0,0], b[1,0], b[2,1]],
                             [b[0,1], b[1,0], b[2,1]],
                             [b[0,1], b[1,0], b[2,0]],
                             [b[0,0], b[1,0], b[2,0]]],

                            [[b[0,0], b[1,0], b[2,1]],
                             [b[0,1], b[1,0], b[2,1]],
                             [b[0,1], b[1,1], b[2,1]],
                             [b[0,0], b[1,1], b[2,1]]],

                            [[b[0,0], b[1,0], b[2,1]],
                             [b[0,0], b[1,1], b[2,1]],
                             [b[0,0], b[1,1], b[2,0]],
                             [b[0,0], b[1,0], b[2,0]]],

                            [[b[0,1], b[1,0], b[2,1]],
                             [b[0,1], b[1,1], b[2,1]],
                             [b[0,1], b[1,1], b[2,0]],
                             [b[0,1], b[1,0], b[2,0]]],

                            [[b[0,0], b[1,1], b[2,1]],
                             [b[0,1], b[1,1], b[2,1]],
                             [b[0,1], b[1,1], b[2,0]],
                             [b[0,0], b[1,1], b[2,0]]],

                            [[b[0,0], b[1,0], b[2,0]],
                             [b[0,1], b[1,0], b[2,0]],
                             [b[0,1], b[1,1], b[2,0]],
                             [b[0,0], b[1,1], b[2,0]]]])

        self._patch_windows['3D'].set_verts(verts)

        return True


## -------------------------------------------------------------------------------------------------
    def _update_plot_nd(self, p_settings:PlotSettings, p_instances:InstDict, **p_kwargs) -> bool:
        """
        The n-dimensional representation of the ring buffer visualizes the removal of obsolete data 
        from the buffer by hiding it behind a semi-transparent rectangle. The visualization starts 
        when the buffer is completely filled and data is removed.

        Parameters
        ----------
        p_settings : PlotSettings
            Object with further plot settings.
        p_instances : InstDict
            Stream instances to be plotted.
        p_kwargs : dict
            Further optional plot parameters.

        Returns
        -------
        bool   
            True, if changes on the plot require a refresh of the figure. False otherwise.          
        """

        # 1 No visualization until the buffer has been filledd
        if ( not self._buffer_full ) or ( len(p_instances) == 0 ): return False


        # 2 Check if the rectangle patches are already created
        if self._patch_windows is None:
            self._patch_windows = {}

            bg = 'lightgrey' #p_settings.axes.get_facecolor()
            ec = p_settings.axes.patch.get_edgecolor()
            obs_window = Rectangle((0,0), 0,0, facecolor = bg, edgecolor=ec, lw = 1, zorder=9999, alpha = 0.75 )
            p_settings.axes.add_patch(obs_window)
            self._patch_windows['nD'] = obs_window


        # 3 Add the hiding plot around obsolete data
        inst_oldest = self._buffer[self._buffer_pos]
        x_limits = p_settings.axes.get_xlim()
        y_limits = p_settings.axes.get_ylim()

        if isinstance( inst_oldest.tstamp, (int, float) ):
            w = inst_oldest.tstamp - x_limits[0]
        else:
            w = mdates.date2num(inst_oldest.tstamp) - x_limits[0]

        h = y_limits[1] - y_limits[0]
        self._patch_windows['nD'].set_bounds(x_limits[0], y_limits[0], w, h)
        self._patch_windows['nD'].set_visible(True)

        return True
