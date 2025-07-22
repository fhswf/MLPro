## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - The integrative middleware framework for standardized machine learning
## -- Package : mlpro.bf.plot
## -- Module  : dataplotting.py
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
## -- 2024-12-10  3.0.0     DA       Created new module dataplotting.py for class DataPlotting
## -------------------------------------------------------------------------------------------------

"""
Ver. 3.0.0 (2024-12-10)

This module provides various classes related to data plotting.

"""


from operator import mod
import numpy as np
from sys import platform

try:
    import matplotlib.pyplot as plt
except:
    pass

import os
import statistics
from mlpro.bf.various import Persistent
from mlpro.bf.data import DataStoring



# Export list for public API
__all__ = [ 'DataPlotting' ]




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