## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - A Synoptic Framework for Standardized Machine Learning Tasks
## -- Package : mlpro.bf
## -- Module  : plot.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2021-10-06  1.0.0     DA       Creation and transfer of classes DataPlotting, Plottable from 
## --                                mlpro.bf.various
## -- 2021-10-25  1.0.1     SY       Improve get_plots() functionality, enable episodic plots
## -- 2021-12-10  1.0.2     SY       Add errors and exceptions, if p_printing is None.
## --                                Clean code assurance.
## -- 2022-10-24  2.0.0     DA       New class PlotSettings and extensions on class Plottable
## -- 2022-10-28  2.0.1     DA       Corrections of class documentations
## -- 2022-10-29  2.0.2     DA       Refactoring of class Plottable
## -- 2022-10-31  2.1.0     DA       Class Plottable: fixes and improvements
## -- 2022-11-07  2.2.0     DA       Class Plottable: new method get_visualization()
## -- 2022-11-09  2.2.1     DA       Classes Plottable, PlotSettings: correction
## -- 2022-11-17  2.3.0     DA       Classes Plottable, PlotSettings: extensions, corrections
## -- 2022-11-18  2.3.1     DA       Classes Plottable, PlotSettings: improvements try/except
## -- 2022-12-20  2.4.0     DA       New method Plottable.set_plot_settings()
## -- 2022-12-28  2.5.0     DA       - Corrections in method Plottable.init_plot()
## --                                - Reduction to one active plot view per task
## -- 2022-12-29  2.6.0     DA       Refactoring of plot settings
## -- 2023-01-01  2.7.0     DA       Class Plottable: introduction of update step rate
## -- 2023-01-04  2.8.0     DA       Class PlotSettings: new parameters p_horizon, p_force_fg
## -- 2023-02-02  2.8.1     MRD      Disable Tkinter backend for macos https://bugs.python.org/issue46573
## -- 2023-02-17  2.8.2     SY       Add p_window_type in DataPlotting
## -- 2023-02-23  2.9.0     DA       Class PlotSettings: new parameter p_view_autoselect
## -- 2023-04-10  2.9.1     MRD      Turn on Tkinter backend for macos
## -- 2023-05-01  2.9.2     DA       Turn off Tkinter backend for macos due to workflow problems
## -- 2023-12-28  2.10.0    DA       Method Plottable._init_plot_3d(): init 3D view perspective
## -- 2024-02-23  2.11.0    DA       Class Plottable: new methods
## --                                - _remove_plot_2d(), _remove_plot_3d(), _remove_plot_nd()
## --                                - __del__() 
## -- 2024-02-24  2.11.1    DA       Class Plottable:
## --                                - new methods remove_plot(), _remove_plot()
## --                                - new methods refresh_plot(), _refresh_plot()
## -- 2024-05-21  2.12.0    DA       Class PlotSettings:
## --                                - parameter horizon replaced by plot_horizon with new default 
## --                                  value 500
## --                                - new parameter data_horizon with default value 1000
## -- 2024-05-22  2.13.0    DA       New method PlotSettings.copy()
## -- 2024-06-04  2.13.1    DA/SK    Turned on TKAgg for Mac
## -- 2024-06-07  2.13.2    SY       Introducing new data plotting type of Episodic Sum
## -- 2024-06-24  2.14.0    DA       New auto-managed attribute Plottable._plot_first_time : bool
## -- 2024-06-25  2.15.0    DA       Class Plottable:
## --                                - removed method set_plot_detail_level()
## --                                - added methods assign_plot_detail_level(), 
## --                                  get_plot_detail_level() and related property plot_detail_level
## --                                - added new constant attribute C_PLOT_DETAIL_LEVEL
## -- 2024-06-26  2.16.0    DA       - Refactoring, corrections, adjustments
## --                                - New property Plottable.color
## --                                - Class PlotSettings: removed parameter p_plot_depth
## -- 2024-07-08  2.16.1    SY       Add MinVal for undefined range in DataPlotting
## -- 2024-10-30  2.17.0    DA       - Class PlotSettings: new methods register(), unregister(),
## --                                  is_last_registered()
## --                                - Class Plottable: extensions on init_plot(), update_plot() 
## --                                - Refactoring: removed par p_force from Plottable.refresh()
## -- 2024-11-10  2.18.0    DA       Bugfix in method Plottable.force_fg()
## -------------------------------------------------------------------------------------------------

"""
Ver. 2.18.0 (2024-11-10)

This module provides various classes related to data plotting.

"""


from operator import mod
import numpy as np
from sys import platform

try:
    from tkinter import *
    import matplotlib
    # Due to bug in TKinter for macos, disabled for macos https://bugs.python.org/issue46573
    #if platform != 'darwin':
    matplotlib.use('TkAgg')
except:
    print('Please install tkinter for a better plot experience')
    import matplotlib

from matplotlib.figure import Figure
from matplotlib.axes import Axes
import matplotlib.pyplot as plt
import os
import statistics
from mlpro.bf.exceptions import ImplementationError, ParamError
from mlpro.bf.various import Persistent
from mlpro.bf.data import DataStoring




## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class PlotSettings:
    """
    Class to specify the context of a subplot.

    Parameters
    ----------
    p_view : str
        ID of the view (see constants C_VIEW_*)
    p_axes : Axes 
        Optional Matplotlib Axes object as destination for plot outputs. Default is None.
    p_pos_x : int 
        Optional x position of a subplot within a Matplotlib figure. Default = 1.
    p_pos_y : int 
        Optional y position of a subplot within a Matplotlib figure. Default = 1.
    p_size_x : int
        Relative size factor in x direction. Default = 1.
    p_size_y : int
        Relative size factor in y direction. Default = 1.
    p_step_rate : int 
        Optional step rate. Decides after how many calls of the update_plot() method the custom 
        methods _update_plot() carries out an output. Default = 1.
    p_plot_horizon : int
        Optional plot horizon for ND plot. A value > 0 limits the number of data entities shown
        in the plot. Default = 500.
    p_data_horizon : int
        Optional data horizon for ND plot. A value > 0 limits the number of data entities buffered 
        internally for plotting. Default = 1000.
    p_detail_level : int 
        Optional detail level. Default = 0.
    p_force_fg : bool
        Optional boolean flag. If True, the releated window is kept in foreground. Default = True.
    p_id : int
        Optional unique id of the subplot within the figure. Default = 1.
    p_view_autoselect : bool
        If True, the final view is automatically selected during runtime. Default = True.
    p_kwargs : dict
        Further optional named parameters.
    """

    C_VIEW_2D   = '2D'
    C_VIEW_3D   = '3D'
    C_VIEW_ND   = 'ND'

    C_VALID_VIEWS   = [ C_VIEW_2D, C_VIEW_3D, C_VIEW_ND ]

## -------------------------------------------------------------------------------------------------
    def __init__( self, 
                  p_view : str, 
                  p_axes : Axes = None, 
                  p_pos_x : int = 1, 
                  p_pos_y : int = 1, 
                  p_size_x : int = 1,
                  p_size_y : int = 1,
                  p_step_rate : int = 1,
                  p_plot_horizon : int = 500,
                  p_data_horizon : int = 1000,
                  p_detail_level : int = 0,
                  p_force_fg : bool = True,
                  p_id : int = 1,
                  p_view_autoselect : bool = True,
                  **p_kwargs ):

        if p_view not in self.C_VALID_VIEWS:
            raise ParamError('Wrong value for parameter p_view. See class mlpro.bf.plot.SubPlotSettings for more details.')

        self.view               = p_view
        self.axes               = p_axes
        self.pos_x              = p_pos_x
        self.pos_y              = p_pos_y
        self.size_x             = p_size_x
        self.size_y             = p_size_y
        self.step_rate          = p_step_rate
        self.detail_level       = p_detail_level
        self.force_fg           = p_force_fg
        self.id                 = p_id
        self.view_autoselect    = p_view_autoselect
        self.kwargs             = p_kwargs.copy()
        self._registered_obj    = []
        self._plot_step_counter = 0

        if ( p_plot_horizon > 0 ) and ( p_data_horizon > 0 ):
            self.plot_horizon = min(p_plot_horizon, p_data_horizon)
            self.data_horizon = max(p_plot_horizon, p_data_horizon)
        else:
            self.plot_horizon    = p_plot_horizon
            self.data_horizon    = p_data_horizon

        
## -------------------------------------------------------------------------------------------------
    def register( self, p_plot_obj : type):
        """
        Registers the specified plotting object. Internally used in class Plottable.

        Parameters
        ----------
        p_plot_obj : type
            Plotting object to be registered
        """

        self._registered_obj.append(p_plot_obj)


## -------------------------------------------------------------------------------------------------
    def unregister( self, p_plot_obj : type):
        """
        Unregisters the specified plotting object. Internally used in class Plottable.

        Parameters
        ----------
        p_plot_obj : type
            Plotting object to be registered
        """

        try:
            self._registered_obj.remove(p_plot_obj)
        except:
            pass


## -------------------------------------------------------------------------------------------------
    def is_last_registered( self, p_plot_obj : type ) -> bool:
        """
        Checks whether the specified plot object was the last one registered. Internally used in
        class Plottable.

        Parameters
        ----------
        p_plot_obj : type
            Plotting object to be registered

        Returns
        -------
        bool
            True, if the specified plotting object was the last one registering. False otherwise.     
        """
        
        try:
            return p_plot_obj == self._registered_obj[-1]
        except:
            return False


## -------------------------------------------------------------------------------------------------
    def copy(self):
        """
        Creates a copy of ifself. The values of following attributes are NOT taken over:
        self._registered_obj, self.plot_step_counter.
        """

        return self.__class__( p_view = self.view,
                                    p_axes = self.axes,
                                    p_pos_x = self.pos_x,
                                    p_pos_y = self.pos_y,
                                    p_size_x = self.size_x,
                                    p_size_y = self.size_y,
                                    p_step_rate = self.step_rate,
                                    p_plot_horizon = self.plot_horizon,
                                    p_data_horizon = self.data_horizon,
                                    p_detail_level = self.detail_level,
                                    p_force_fg = self.force_fg,
                                    p_id = self.id,
                                    p_view_autoselect = self.view_autoselect,
                                    p_kwargs = self.kwargs )

    




## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class Plottable:
    """
    Property class that inherits the ability to be plottable. The class is prepared for plotting with
    MatPlotLib but not restricted to it. Three different views are supported:
    
    2D: 2-dimensional plot
    3D: 3-dimensional plot
    ND: Multidimensional plot

    See class Plotsettings for further detais.

    Parameters
    ----------
    p_visualize : bool
        Boolean switch for visualisation. Default = False.

    Attributes
    ----------
    C_PLOT_ACTIVE : bool
        Custom attribute to turn on or off the plot functionality. Must be turned on explicitely.
    C_PLOT_STANDALONE : bool = True
        Custom attribute to be set to True, if the plot needs a separate subplot or False if the 
        plot can be added to an existing subplot.
    C_PLOT_VALID_VIEWS : list = [PlotSettings]
        Custom list of views that are supported/implemented (see class PlotSettings)
    C_PLOT_DEFAULT_VIEW : str = ''
        Custom attribute for the default view. See class PlotSettings for more details.
    C_PLOT_DETAIL_LEVEL : int = 0
        Custom attribute for the assigned detail level. See method assign_plot_detail_level() for
        more details.
    color : str
        Plot color. See also: https://matplotlib.org/stable/gallery/color/named_colors.html
    plot_detail_level : int
        Own plot detail level.
    """

    C_PLOT_ACTIVE : bool        = False
    C_PLOT_STANDALONE : bool    = True
    C_PLOT_VALID_VIEWS : list   = []
    C_PLOT_DEFAULT_VIEW : str   = PlotSettings.C_VIEW_ND
    C_PLOT_DETAIL_LEVEL : int   = 0 

## -------------------------------------------------------------------------------------------------
    def __init__(self, p_visualize:bool=False):
        self._visualize                    = self.C_PLOT_ACTIVE and p_visualize
        self._plot_settings : PlotSettings = None
        self.plot_detail_level             = self.C_PLOT_DETAIL_LEVEL
        self._plot_initialized : bool      = False
        self._plot_first_time : bool       = True
        self._plot_own_figure : bool       = False
        self._plot_color                   = None


## -------------------------------------------------------------------------------------------------
    def get_plot_settings(self) -> PlotSettings:
        return self._plot_settings


## -------------------------------------------------------------------------------------------------
    def set_plot_settings(self, p_plot_settings : PlotSettings ):
        """
        Sets plot settings in advance (before initialization of plot).

        Parameters
        ----------
        p_plot_settings : PlotSettings
            New PlotSettings to be set. If None, the default view is plotted (see attribute 
            C_PLOT_DEFAULT_VIEW).
        """

        try:
            if self._plot_initialized: return
        except:
            pass

        if p_plot_settings is not None: 
            self._plot_settings = p_plot_settings
        elif self._plot_settings is None:
            try:
                self._plot_settings = PlotSettings(p_view=self.C_PLOT_DEFAULT_VIEW)
            except ParamError:
                # Plot functionality turned on but not implemented
                raise ImplementationError('Please check attribute C_PLOT_DEFAULT_VIEW')    

        self._plot_step_counter = 0
   

## -------------------------------------------------------------------------------------------------
    def init_plot( self, 
                   p_figure:Figure = None,
                   p_plot_settings : PlotSettings = None ) -> bool:
        """
        Initializes the plot functionalities of the class.

        Parameters
        ----------
        p_figure : Matplotlib.figure.Figure, optional
            Optional MatPlotLib host figure, where the plot shall be embedded. The default is None.
        p_plot_settings : PlotSettings
            Optional plot settings. If None, the default view is plotted (see attribute C_PLOT_DEFAULT_VIEW).

        """

        # 1 Plot functionality turned on? Initialization already called?
        try:
            if ( not self.C_PLOT_ACTIVE ) or ( not self._visualize ): return
        except:
            return

        try:
            if ( p_plot_settings.detail_level > 0 ) and ( p_plot_settings.detail_level < self.plot_detail_level ): return
        except:
            self.plot_detail_level = self.C_PLOT_DETAIL_LEVEL

        try:
            if self._plot_initialized: return
        except:
            self._plot_own_figure   = False

        plt.ioff()


        # 2 Prepare internal data structures

        # 2.1 Dictionary with methods for initialization and update of a plot per view 
        self._plot_methods = { PlotSettings.C_VIEW_2D : ( self._init_plot_2d, self._update_plot_2d, self._remove_plot_2d ), 
                               PlotSettings.C_VIEW_3D : ( self._init_plot_3d, self._update_plot_3d, self._remove_plot_3d ), 
                               PlotSettings.C_VIEW_ND : ( self._init_plot_nd, self._update_plot_nd, self._remove_plot_nd ) }

        # 2.2 Plot settings per view
        self.set_plot_settings( p_plot_settings=p_plot_settings )
        
        # 2.3 Setup the Matplotlib host figure if no one is provided as parameter
        if p_figure is None:
            self._figure : Figure   = self._init_figure()
            self._plot_own_figure   = True
        else:
            self._figure : Figure   = p_figure
            
        self._plot_settings.register( p_plot_obj = self )

            
        # 3 Call of all initialization methods of the required views
        view = self._plot_settings.view
        try:
            plot_method = self._plot_methods[view][0]
        except KeyError:
            raise ParamError('Parameter p_plot_settings: wrong view "' + str(view) + '"')

        plot_method(p_figure=self._figure, p_settings=self._plot_settings)

        if self._plot_settings.axes is None:
            raise ImplementationError('Please set attribute "axes" in your custom _init_plot_' + view + ' method')
                

        # 4 In standalone mode: refresh figure
        if self._plot_own_figure:
            self._figure.canvas.draw()
            self._figure.canvas.flush_events()


        # 5 Marker to ensure that initialization runs only once
        self._plot_initialized = True
        self._plot_first_time  = True


## -------------------------------------------------------------------------------------------------
    def get_visualization(self) -> bool:
        return self._visualize


## -------------------------------------------------------------------------------------------------
    def set_plot_step_rate(self, p_step_rate:int):
        if p_step_rate > 0: self._plot_settings.step_rate = p_step_rate


## -------------------------------------------------------------------------------------------------
    def get_plot_color(self):
        try:
            return self._plot_color
        except:
            self._plot_color = None
            return self._plot_color

    
## -------------------------------------------------------------------------------------------------
    def set_plot_color(self, p_color : str):
        self._plot_color = p_color


## -------------------------------------------------------------------------------------------------
    def get_plot_detail_level(self) -> int:
        try:
            return self._plot_detail_level
        except:
            self.assign_plot_detail_level( p_detail_level = self.C_PLOT_DETAIL_LEVEL )
            return self._plot_detail_level


## -------------------------------------------------------------------------------------------------
    def assign_plot_detail_level(self, p_detail_level:int):
        """
        Assigns an own plot detail level. Plots are carried out only, if the specified detail level
        is less or equal to self._plot_settings.detail_level or self._plot_settings.detail_level = 0.

        Parameters
        ----------
        p_detail_level : int
            Integer detail level >=0 to be assigned.
        """

        self._plot_detail_level = max(0, p_detail_level)


## -------------------------------------------------------------------------------------------------
    def _init_figure(self) -> Figure:
        """
        Custom method to initialize a suitable standalone Matplotlib figure.

        Returns
        -------
        figure : Matplotlib.figure.Figure
            Matplotlib figure object to host the subplot(s)
        """

        fig = plt.figure()   
        plt.show(block=False)
        self._force_fg(p_fig=fig)
        return fig


## -------------------------------------------------------------------------------------------------
    def force_fg(self):
        """
        Internal use.
        """

        # 1 Plot functionality turned on?
        try:
            if ( not self.C_PLOT_ACTIVE ) or ( not self._visualize ): return
        except:
            return
        
        # 2 Call internal custom method
        self._force_fg(p_fig = self._figure)


## -------------------------------------------------------------------------------------------------
    def _force_fg(self, p_fig : Figure):
        """
        Internal use.
        """

        if not self._plot_settings.force_fg: return

        backend = matplotlib.get_backend()

        if backend == 'TkAgg':
            p_fig.canvas.manager.window.attributes('-topmost', True)        


## -------------------------------------------------------------------------------------------------
    def refresh_plot(self):
        """
        Refreshes the plot.
        """
    
        # 1 Plot functionality turned on?
        try:
            if ( not self.C_PLOT_ACTIVE ) or ( not self._visualize ): return
        except:
            return
        

         # 2 Update the plot step counter
        self._plot_settings._plot_step_counter = mod(self._plot_settings._plot_step_counter + 1, self._plot_settings.step_rate)
        

        # 3 Refresh plot
        if self._plot_settings._plot_step_counter == 0: self._refresh_plot()
            

## -------------------------------------------------------------------------------------------------
    def _refresh_plot(self):
        """
        Custom method to refresh the plot. Default implementation assumes standard use of Matplotlib.
        """

        self._figure.canvas.draw()
        self._figure.canvas.flush_events()
    

## -------------------------------------------------------------------------------------------------
    def _init_plot_2d(self, p_figure:Figure, p_settings:PlotSettings):
        """
        Custom method to initialize a 2D plot. If attribute p_settings.axes is not None the 
        initialization shall be done there. Otherwise a new MatPlotLib Axes object shall be 
        created in the given figure and stored in p_settings.axes.

        Note: Please call this method in your custom implementation to create a default subplot.

        Parameters
        ----------
        p_figure : Matplotlib.figure.Figure
            Matplotlib figure object to host the subplot(s).
        p_settings : PlotSettings
            Object with further plot settings.
        """

        if p_settings.axes is None:
            p_settings.axes = p_figure.add_subplot( p_settings.pos_y, p_settings.pos_x, p_settings.id )


## -------------------------------------------------------------------------------------------------
    def _init_plot_3d(self, p_figure:Figure, p_settings:PlotSettings):
        """
        Custom method to initialize a 3D plot. If attribute p_settings.axes is not None the 
        initialization shall be done there. Otherwise a new MatPlotLib Axes object shall be 
        created in the given figure and stored in p_settings.axes.

        Note: Please call this method in your custom implementation to create a default subplot.

        Parameters
        ----------
        p_figure : Matplotlib.figure.Figure
            Matplotlib figure object to host the subplot(s).
        p_settings : PlotSettings
            Object with further plot settings.
        """

        if p_settings.axes is None:
            p_settings.axes = p_figure.add_subplot( p_settings.pos_y, p_settings.pos_x, p_settings.id, projection='3d' )
            p_settings.axes.set_proj_type(proj_type='persp', focal_length=0.3)


## -------------------------------------------------------------------------------------------------
    def _init_plot_nd(self, p_figure:Figure, p_settings:PlotSettings):
        """
        Custom method to initialize a nD plot. If attribute p_settings.axes is not None the 
        initialization shall be done there. Otherwise a new MatPlotLib Axes object shall be 
        created in the given figure and stored in p_settings.axes.

        Note: Please call this method in your custom implementation to create a default subplot.

        Parameters
        ----------
        p_figure : Matplotlib.figure.Figure
            Matplotlib figure object to host the subplot(s).
        p_settings : PlotSettings
            Object with further plot settings.
        """

        if p_settings.axes is None:
            p_settings.axes = p_figure.add_subplot( p_settings.pos_y, p_settings.pos_x, p_settings.id )


## -------------------------------------------------------------------------------------------------
    def update_plot(self, **p_kwargs):
        """
        Updates the plot.

        Parameters
        ----------
        **p_kwargs
            Implementation-specific plot data and/or parameters.
        """

        # 0 Plot functionality turned on?
        try:
            if ( not self.C_PLOT_ACTIVE ) or ( not self._visualize ): return
        except:
            return
            

         # 1 Plot already initialized?
        try:
            if not self._plot_initialized: self.init_plot()
        except: 
            self.init_plot()


        # 2 Check the assigned/required detail level
        if ( self._plot_settings.detail_level > 0 ) and ( self._plot_settings.detail_level < self.plot_detail_level ): return


        # 3 Call of all required plot methods
        view = self._plot_settings.view
        self._plot_methods[view][1](p_settings=self._plot_settings, **p_kwargs)

        
        # 4 The last plotting object for the figure refreshs the plot
        if self._plot_settings.is_last_registered( p_plot_obj = self ):
            self.refresh_plot()


## -------------------------------------------------------------------------------------------------
    def _update_plot_2d(self, p_settings:PlotSettings, **p_kwargs):
        """
        Custom method to update the 2d plot. The related MatPlotLib Axes object is stored in p_settings.

        Parameters
        ----------
        p_settings : PlotSettings
            Object with further plot settings.
        **p_kwargs 
            Implementation-specific data and parameters.             
        """

        pass


## -------------------------------------------------------------------------------------------------
    def _update_plot_3d(self, p_settings:PlotSettings, **p_kwargs):
        """
        Custom method to update the 3d plot. The related MatPlotLib Axes object is stored in p_settings.

        Parameters
        ----------
        p_settings : PlotSettings
            Object with further plot settings.
        **p_kwargs 
            Implementation-specific data and parameters.             
        """

        pass


## -------------------------------------------------------------------------------------------------
    def _update_plot_nd(self, p_settings:PlotSettings, **p_kwargs):
        """
        Custom method to update the nd plot. The related MatPlotLib Axes object is stored in p_settings.

        Parameters
        ----------
        p_settings : PlotSettings
            Object with further plot settings.
        **p_kwargs 
            Implementation-specific data and parameters.             
        """

        pass


## -------------------------------------------------------------------------------------------------
    def remove_plot(self, p_refresh:bool = True):
        """"
        Removes the plot and optionally refreshes the display.

        Parameters
        ----------
        p_refresh : bool = True
            On True the display is refreshed after removal
        """

        # 1 Plot functionality turned on?
        try:
            if not self._plot_initialized: return
        except:
            return
            
        # 2 Call _remove_plot method of current view
        view = self._plot_settings.view
        self._plot_methods[view][2]()

        # 3 Optionally refresh
        if p_refresh: self.refresh_plot()

        # 4 Clear internal plot parameters
        self._plot_settings.unregister( p_plot_obj = self )
        self._plot_first_time = True
   

## -------------------------------------------------------------------------------------------------
    def _remove_plot_2d(self):
        """
        Custom method to remove 2D plot artifacts when object is destroyed.
        """

        pass


## -------------------------------------------------------------------------------------------------
    def _remove_plot_3d(self):
        """
        Custom method to remove 3D plot artifacts when object is destroyed.
        """

        pass


## -------------------------------------------------------------------------------------------------
    def _remove_plot_nd(self):
        """
        Custom method to remove nd plot artifacts when object is destroyed.
        """

        pass


## -------------------------------------------------------------------------------------------------
    color             = property( fget = get_plot_color, fset = set_plot_color )
    plot_detail_level = property( fget = get_plot_detail_level, fset = assign_plot_detail_level )





## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class DataPlotting(Persistent):
    """
    This class provides a functionality to plot the stored values of variables.

    Parameters
    ----------
    p_data : DataStoring
        Data object with stored variables values.
    p_type : str, optional
        Type of plot. The default is C_PLOT_TYPE_EP.
    p_window : int, optional
        Moving average parameter. The default is 100.
    p_showing : Bool, optional
        Showing graphs after they are generated. The default is True.
    p_printing : dict, optional
        Additional important parameters for plotting.
        [0] = Bool : Whether the stored values is plotted.
        [1] = Float : Min. value on graph.
        [2] = Float : Max. value on graph. Set to -1, if you want to set min/max value according to the stored values.
        Example = {"p_variable_1" : [True,0,-1],
                   "p_variable_2" : [True,-0.5,10]}.
        The default is None.
    p_figsize : int, optional
        Frame size. The default is (7,7).
    p_color : str, optional
        Line colors. The default is "darkblue".
    p_window_type : str, optional
        Plotting type for moving average. The default is 'same'. Options: 'same', 'full', 'valid'
        
    Attributes
    ----------
    C_PLOT_TYPE_CY : str
        one of the plotting types, which plot the graph with multiple lines according to the number of frames.
    C_PLOT_TYPE_EP : str
        one of the plotting types, which plot the graph everything in one line regardless the number of frames.
    C_PLOT_TYPE_EP_M : str
        one of the plotting types, which plot only the mean value of each variable for each frame.
        
    """

    C_PLOT_TYPE_CY = 'Cyclic'
    C_PLOT_TYPE_EP = 'Episodic'
    C_PLOT_TYPE_EP_M = 'Episodic Mean'
    C_PLOT_TYPE_EP_S = 'Episodic Sum'

## -------------------------------------------------------------------------------------------------
    def __init__(self, p_data: DataStoring, p_type=C_PLOT_TYPE_EP, p_window=100,
                 p_showing=True, p_printing=None, p_figsize=(7, 7), p_color="darkblue",
                 p_window_type='same'):
        self.data = p_data
        self.type = p_type
        self.window = p_window
        self.showing = p_showing
        self.plots = [[], []]
        self.printing = p_printing
        self.figsize = p_figsize
        self.color = p_color
        self.window_type = p_window_type


## -------------------------------------------------------------------------------------------------
    def get_plots(self):
        """
        A function to plot data.
        """

        if self.type == 'Cyclic':
            self.plots_type_cy()
        elif self.type == 'Episodic':
            self.plots_type_ep()
        elif self.type == 'Episodic Mean':
            self.plots_type_ep_mean()
        elif self.type == 'Episodic Sum':
            self.plots_type_ep_sum()


## -------------------------------------------------------------------------------------------------
    def plots_type_cy(self):
        """
        A function to plot data per cycle.
        """

        for name in self.data.names:
            maxval = 0
            minval = 0
            try:
                if self.printing[name][0]:
                    fig = plt.figure(figsize=self.figsize)
                    lines = []
                    label = []
                    plt.title(name)
                    plt.grid(True, which="both", axis="both")
                    for fr in range(len(self.data.memory_dict[name])):
                        fr_id = self.data.frame_id[name][fr]
                        lines += plt.plot(self.moving_mean(self.data.get_values(name, fr_id), self.window),
                                          color=self.color, alpha=(fr + 1.0) / (len(self.data.memory_dict[name]) + 1))
                        if self.printing[name][2] == -1:
                            maxval = max(max(self.data.get_values(name, fr_id)), maxval)
                            minval = min(min(self.data.get_values(name, fr_id)), minval)
                        else:
                            maxval = self.printing[name][2]
                            minval = self.printing[name][1]
                        label.append("%s" % fr_id)
                    plt.ylim(minval, maxval)
                    plt.xlabel("cycles")
                    plt.legend(label, bbox_to_anchor=(1, 0.5), loc="center left")
                    self.plots[0].append(name)
                    self.plots[1].append(fig)
                    if self.showing:
                        plt.show()
                    else:
                        plt.close(fig)
            except:
                pass


## -------------------------------------------------------------------------------------------------
    def plots_type_ep(self):
        """
        A function to plot data per frame by extending the cyclic plots in one plot.
        """

        for name in self.data.names:
            maxval = 0
            minval = 0
            try:
                if self.printing[name][0]:
                    fig = plt.figure(figsize=self.figsize)
                    lines = []
                    data = []
                    plt.title(name)
                    plt.grid(True, which="both", axis="both")
                    for fr in range(len(self.data.memory_dict[name])):
                        fr_id = self.data.frame_id[name][fr]
                        data.extend(self.data.get_values(name, fr_id))
                        if self.printing[name][2] == -1:
                            maxval = max(max(self.data.get_values(name, fr_id)), maxval)
                            minval = min(min(self.data.get_values(name, fr_id)), minval)
                        else:
                            maxval = self.printing[name][2]
                            minval = self.printing[name][1]
                    lines += plt.plot(self.moving_mean(data[:], self.window), color=self.color)
                    plt.ylim(minval, maxval)
                    plt.xlabel("continuous cycles")
                    self.plots[0].append(name)
                    self.plots[1].append(fig)
                    if self.showing:
                        plt.show()
                    else:
                        plt.close(fig)
            except:
                pass


## -------------------------------------------------------------------------------------------------
    def plots_type_ep_mean(self):
        """
        A function to plot data per frame according to its mean value.
        """

        for name in self.data.names:
            maxval = 0
            minval = 0
            try:
                if self.printing[name][0]:
                    fig = plt.figure(figsize=self.figsize)
                    lines = []
                    data = []
                    plt.title(name)
                    plt.grid(True, which="both", axis="both")
                    for fr in range(len(self.data.memory_dict[name])):
                        fr_id = self.data.frame_id[name][fr]
                        data.extend([statistics.mean(self.data.get_values(name, fr_id))])
                    if self.printing[name][2] == -1:
                        maxval = max(max(data[:]), maxval)
                        minval = min(min(data[:]), minval)
                    else:
                        maxval = self.printing[name][2]
                        minval = self.printing[name][1]
                    lines += plt.plot(self.moving_mean(data[:], self.window), color=self.color)
                    plt.ylim(minval, maxval)
                    plt.xlabel("episodes")
                    self.plots[0].append(name)
                    self.plots[1].append(fig)
                    if self.showing:
                        plt.show()
                    else:
                        plt.close(fig)
            except:
                pass


## -------------------------------------------------------------------------------------------------
    def plots_type_ep_sum(self):
        """
        A function to plot data per frame according to its sum value.
        """

        for name in self.data.names:
            maxval = 0
            minval = 0
            try:
                if self.printing[name][0]:
                    fig = plt.figure(figsize=self.figsize)
                    lines = []
                    data = []
                    plt.title(name)
                    plt.grid(True, which="both", axis="both")
                    for fr in range(len(self.data.memory_dict[name])):
                        fr_id = self.data.frame_id[name][fr]
                        data.extend([sum(self.data.get_values(name, fr_id))])
                    if self.printing[name][2] == -1:
                        maxval = max(max(data[:]), maxval)
                        minval = min(min(data[:]), minval)
                    else:
                        maxval = self.printing[name][2]
                        minval = self.printing[name][1]
                    lines += plt.plot(self.moving_mean(data[:], self.window), color=self.color)
                    plt.ylim(minval, maxval)
                    plt.xlabel("episodes")
                    self.plots[0].append(name)
                    self.plots[1].append(fig)
                    if self.showing:
                        plt.show()
                    else:
                        plt.close(fig)
            except:
                pass


## -------------------------------------------------------------------------------------------------
    def moving_mean(self, p_inputs, p_window):
        """
        This method creates a series of averages of different subsets of the full data set.

        Parameters
        ----------
        p_inputs : list of floats
            input dataset.
        p_window : int
            moving average parameter.

        Returns
        -------
        outputs : list of floats
            transformed data set.

        """

        inputs = np.array(p_inputs)
        outputs = np.zeros_like(inputs)
        if len(inputs.shape) == 1:
            outputs = np.convolve(inputs, np.ones((p_window,)) / p_window, mode=self.window_type)
        else:
            for col in range(inputs.shape[1]):
                outputs[:, col] = np.convolve(inputs[:, col], np.ones((p_window,)) / p_window, mode=self.window_type)
        return outputs


## -------------------------------------------------------------------------------------------------
    def save_plots(self, p_path, p_format, p_dpi_mul=1):
        """
        This method is used to save generated plots.

        Parameters
        ----------
        p_path : str
            Path where file will be saved.
        p_format : str
            Format of the saved file.
            Options: 'eps', 'jpg', 'png', 'pdf', 'svg'.
        p_dpi_mul : int, optional
            Saving plots parameter. The default is 1.

        Returns
        -------
        bool
            True, if plots where saved successfully. False otherwise..

        """

        num_plots = len(self.plots[0])
        if num_plots == 0: return False

        try:
            if not os.path.exists(p_path):
                os.makedirs(p_path)
            for idx in range(num_plots):
                self.plots[1][idx].savefig(p_path + os.sep + self.plots[0][idx] + "." + p_format, dpi=500 * p_dpi_mul,
                                           bbox_inches='tight')
            return True
        except:
            return False