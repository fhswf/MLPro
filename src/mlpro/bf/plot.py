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
## -------------------------------------------------------------------------------------------------

"""
Ver. 2.2.0 (2022-11-07)

This module provides various classes related to data plotting.
"""


from operator import mod
import numpy as np
from matplotlib.figure import Figure
from matplotlib.axes import Axes
import matplotlib.pyplot as plt
import os
import statistics
from mlpro.bf.exceptions import ImplementationError, ParamError
from mlpro.bf.various import LoadSave
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
    p_axes : Axes = None
        Optional Matplotlib Axes object as destination for plot outputs
    p_pos_x : int = 0
        Optional x position of a subplot within a Matplotlib figure
    p_pos_y : int = 0
        Optional y position of a subplot within a Matplotlib figure
    p_kwargs : dict
        Further optional named parameters
    """

    C_VIEW_2D   = '2D'
    C_VIEW_3D   = '3D'
    C_VIEW_ND   = 'ND'

    C_VALID_VIEWS   = [ C_VIEW_2D, C_VIEW_3D, C_VIEW_ND ]

## -------------------------------------------------------------------------------------------------
    def __init__( self, p_view:str, p_axes:Axes=None, p_pos_x:int=0, p_pos_y:int=0, **p_kwargs ):

        if p_view not in self.C_VALID_VIEWS:
            raise ParamError('Wrong value for parameter p_view. See class mlpro.bf.plot.SubPlotSettings for more details.')

        self.view      = p_view.copy()
        self.axes      = p_axes
        self.pos_x     = p_pos_x
        self.pos_y     = p_pos_y
        self.kwargs    = p_kwargs.copy()





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
        Boolean switch for env/agent visualisation. Default = False.

    Attributes
    ----------
    C_PLOT_ACTIVE : bool
        Custom attribute to turn on or off the plot functionality. Must be turned on explicitely.
    C_PLOT_STANDALONE : bool = True
        Custom attribute to be set to True, if the plot needs a separate subplot or False if the 
        plot can be added to an existing subplot.
    C_PLOT_VALID_VIEWS : list = []
        Custom list of views that are supported/implemented (see class PlotSettings)
    C_PLOT_DEFAULT_VIEW : str = ''
        Custom attribute for the default view. See class PlotSettings for more details.
    """

    C_PLOT_ACTIVE : bool        = False
    C_PLOT_STANDALONE : bool    = True
    C_PLOT_VALID_VIEWS : list   = []
    C_PLOT_DEFAULT_VIEW : str   = PlotSettings.C_VIEW_ND

## -------------------------------------------------------------------------------------------------
    def __init__(self, p_visualize:bool=False):
        self._visualize = self.C_PLOT_ACTIVE and p_visualize


## -------------------------------------------------------------------------------------------------
    def init_plot( self, 
                   p_figure:Figure=None,
                   p_plot_settings:list=[],
                   p_plot_depth:int=0,
                   p_detail_level:int=0,
                   p_step_rate:int=1,
                   **p_kwargs):
        """
        Initializes the plot functionalities of the class.

        Parameters
        ----------
        p_figure : Matplotlib.figure.Figure, optional
            Optional MatPlotLib host figure, where the plot shall be embedded. The default is None.
        p_plot_settings : list
            Optional list of objects of class PlotSettings. All subplots that are addresses in the list
            are plotted in parallel. If the list is empty the default view is plotted (see attribute C_PLOT_DEFAULT_VIEW).
        p_plot_depth : int = 0
            Optional plot depth in case of hierarchical plotting. A value of 0 means that the plot 
            depth is unlimited.
        p_detail_level : int = 0
            Optional detail level.
        p_step_rate : int = 1
            Decides after how many calls of the update_plot() method the custom methods 
            _update_plot() make an output.
        **p_kwargs : dict
            Further optional plot parameters.    
        """

        # 0 Plot functionality turned on? Initialization already called?
        try:
            if not self._visualize: return
        except:
            return

        try:
            self._plot_initialized
            return
        except:
            pass


        # 1 Initialize internal plot attributes
        self._plot_depth        = p_plot_depth
        self._plot_step_counter = 0
        self._plot_kwargs       = p_kwargs.copy()
        self.set_plot_step_rate(p_step_rate)
        self.set_plot_detail_level(p_detail_level=p_detail_level)


        # 2 Prepare internal dictionaries

        # 2.1 Dictionary with plot settings per view
        self._plot_settings = {}
        if len(p_plot_settings)!=0:
            for ps in p_plot_settings:
                self._plot_settings[ps.view] = ps
        else:
            try:
                self._plot_settings[self.C_PLOT_DEFAULT_VIEW] = PlotSettings(p_view=self.C_PLOT_DEFAULT_VIEW)
            except:
                # Plot functionality turned on but not implemented
                raise ImplementationError('Please set attribute C_PLOT_DEFAULT_VIEW')               

        # 2.2 Dictionary with methods for initialization and update of a plot per view 
        self._plot_methods = { PlotSettings.C_VIEW_2D : [ self._init_plot_2d, self._update_plot_2d ], 
                               PlotSettings.C_VIEW_3D : [ self._init_plot_3d, self._update_plot_3d ], 
                               PlotSettings.C_VIEW_ND : [ self._init_plot_nd, self._update_plot_nd ] }


        # 3 Setup the Matplotlib host figure if no one is provided as parameter
        if p_figure is None:
            self._figure : Figure   = self._init_figure()
            self._plot_own_figure   = True
        else:
            self._figure : Figure   = p_figure
            self._plot_own_figure   = False


        # 4 Call of all initialization methods of the required views
        for view in self._plot_settings:
            try:
                self._plot_methods[view][0](p_figure=self._figure, p_settings=self._plot_settings[view])
                if self._plot_settings[view].axes is None:
                    raise ImplementationError('Please set attribute "axes" in your custom _init_plot_' + view + ' method')

            except:
                raise ParamError('Parameter p_plot_settings: wrong view "' + str(view) + '"')


        # 5 Marker to ensure that initialization runs only once
        self._plot_initialized = True


## -------------------------------------------------------------------------------------------------
    def get_visualization(self) -> bool:
        return self._visualize


## -------------------------------------------------------------------------------------------------
    def _init_figure(self) -> Figure:
        """
        Custom method to initialize a suitable standalone Matplotlib figure.

        Returns
        -------
        figure : Matplotlib.figure.Figure
            Matplotlib figure object to host the subplot(s)
        """

        return Figure()            


## -------------------------------------------------------------------------------------------------
    def _init_plot_2d(self, p_figure:Figure, p_settings:PlotSettings):
        """
        Custom method to initialize a 2D plot. If attribute p_settings.axes is not None the 
        initialization shall be done there. Otherwise a new MatPlotLib Axes object shall be 
        created in the given figure and stored in p_settings.axes.

        Parameters
        ----------
        p_figure : Matplotlib.figure.Figure
            Matplotlib figure object to host the subplot(s).
        p_settings : PlotSettings
            Object with further plot settings.
        """

        pass


## -------------------------------------------------------------------------------------------------
    def _init_plot_3d(self, p_figure:Figure, p_settings:PlotSettings):
        """
        Custom method to initialize a 3D plot. If attribute p_settings.axes is not None the 
        initialization shall be done there. Otherwise a new MatPlotLib Axes object shall be 
        created in the given figure and stored in p_settings.axes.

        Parameters
        ----------
        p_figure : Matplotlib.figure.Figure
            Matplotlib figure object to host the subplot(s).
        p_settings : PlotSettings
            Object with further plot settings.
        """

        pass


## -------------------------------------------------------------------------------------------------
    def _init_plot_nd(self, p_figure:Figure, p_settings:PlotSettings):
        """
        Custom method to initialize a nD plot. If attribute p_settings.axes is not None the 
        initialization shall be done there. Otherwise a new MatPlotLib Axes object shall be 
        created in the given figure and stored in p_settings.axes.

        Parameters
        ----------
        p_figure : Matplotlib.figure.Figure
            Matplotlib figure object to host the subplot(s).
        p_settings : PlotSettings
            Object with further plot settings.
        """

        pass


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
            if not self._visualize: return
        except:
            return
            
         # 1 Plot already initiated?
        try:
            self._plot_initialized
        except: 
            self.init_plot()

        # 2 Call of all required plot methods
        for view in self._plot_settings:
            self._plot_methods[view][1](p_settings=self._plot_settings[view], p_kwargs=p_kwargs)

        # 3 Update content of own(!) figure after self._plot_step_rate calls
        if self._plot_own_figure:
            self._plot_step_counter = mod(self._plot_step_counter+1, self._plot_step_rate)
            if self._plot_step_counter==0: 
                self._figure.canvas.draw()
                self._figure.canvas.flush_events()


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
    def set_plot_step_rate(self, p_step_rate:int):
        self._plot_step_rate = min(p_step_rate,1)


## -------------------------------------------------------------------------------------------------
    def set_plot_detail_level(self, p_detail_level:int):
        self._plot_detail_level = min(0, p_detail_level)





## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class DataPlotting(LoadSave):
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

## -------------------------------------------------------------------------------------------------
    def __init__(self, p_data: DataStoring, p_type=C_PLOT_TYPE_EP, p_window=100,
                 p_showing=True, p_printing=None, p_figsize=(7, 7), p_color="darkblue"):
        self.data = p_data
        self.type = p_type
        self.window = p_window
        self.showing = p_showing
        self.plots = [[], []]
        self.printing = p_printing
        self.figsize = p_figsize
        self.color = p_color


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


## -------------------------------------------------------------------------------------------------
    def plots_type_cy(self):
        """
        A function to plot data per cycle.
        """

        for name in self.data.names:
            maxval = 0
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
                        else:
                            maxval = self.printing[name][2]
                        label.append("%s" % fr_id)
                    plt.ylim(self.printing[name][1], maxval)
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
                        else:
                            maxval = self.printing[name][2]
                    lines += plt.plot(self.moving_mean(data[:], self.window), color=self.color)
                    plt.ylim(self.printing[name][1], maxval)
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
                    else:
                        maxval = self.printing[name][2]
                    lines += plt.plot(self.moving_mean(data[:], self.window), color=self.color)
                    plt.ylim(self.printing[name][1], maxval)
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
            outputs = np.convolve(inputs, np.ones((p_window,)) / p_window, mode='same')
        else:
            for col in range(inputs.shape[1]):
                outputs[:, col] = np.convolve(inputs[:, col], np.ones((p_window,)) / p_window, mode='same')
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