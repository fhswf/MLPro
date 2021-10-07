## -------------------------------------------------------------------------------------------------
## -- Project : FH-SWF Automation Technology - Common Code Base (CCB)
## -- Package : mlpro.bf
## -- Module  : plot
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2021-10-06  1.0.0     DA       Creation and transfer of classes DataPlotting, Plottable from 
## --                                mlpro.bf.various
## -------------------------------------------------------------------------------------------------

"""
Ver. 1.0.0 (2021-10-06)

This module provides various classes related to data plotting.
"""


import numpy as np
import matplotlib.pyplot as plt
import os
from mlpro.bf.various import LoadSave
from mlpro.bf.data import DataStoring




## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class Plottable:
    """
    Property class that inherits the ability to be plottable.
    """

## -------------------------------------------------------------------------------------------------
    def init_plot(self, p_figure=None):
        """
        Initializes the plot. Optionally the plot itself will be embedded in a matplotlib figure.

        Parameters:
            p_figure            Optional MatPlotLib host figure, where the plot shall be embedded
        """

        pass


## -------------------------------------------------------------------------------------------------
    def update_plot(self):
        """
        Updates the plot.
        """

        pass





## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class DataPlotting(LoadSave):
    """
    This class provides a functionality to plot the stored values of variables.
    """

## -------------------------------------------------------------------------------------------------
    def __init__(self, p_data:DataStoring, p_window=100, p_showing=True, p_printing=None):
        """
        Parameters:
            p_data        Data object with stored variables values
            p_window      INT : Moving average parameter. default: 100
            p_showing     BOOL : Showing graphs after they are generated
            p_printing    Additional information for plotting at the end of
                          training.
                          [0] = Bool : Whether the stored values is plotted
                          [1] = Float : Min. value on graph
                          [2] = Float : Max. value on graph.
                                Set to -1, if you want to set min/max value
                                according to the stored values.
                          Example = {"p_variable_1" : [True,0,-1],
                                     "p_variable_2" : [True,-0.5,10]}
        """

        self.data       = p_data
        self.window     = p_window
        self.showing    = p_showing
        self.plots      = [[],[]]
        self.printing   = p_printing


## -------------------------------------------------------------------------------------------------
    def get_plots(self):
        """
        A function to plot data
        """
        for name in self.data.names:
            maxval  = 0
            if self.printing[name][0]:
                fig     = plt.figure(figsize=(7,7))
                lines   = []
                label   = []
                plt.title(name)
                plt.grid(True, which="both", axis="both")
                for fr in range(len(self.data.memory_dict[name])):
                    fr_id = self.data.frame_id[name][fr]
                    lines += plt.plot(self.moving_mean(self.data.get_values(name,fr_id),self.window), color="darkblue", alpha=(fr+1.0)/(len(self.data.memory_dict[name])+1))
                    if self.printing[name][2] == -1:
                        maxval = max(max(self.data.get_values(name,fr_id)), maxval)
                    else:
                        maxval = self.printing[name][2]
                    label.append("%s"%fr_id)
                plt.ylim(self.printing[name][1], maxval)
                plt.xlabel("cycles")
                plt.legend(label, bbox_to_anchor = (1,0.5), loc = "center left")
                self.plots[0].append(name)
                self.plots[1].append(fig)
                if self.showing:
                    plt.show()
                else:
                    plt.close(fig)


## -------------------------------------------------------------------------------------------------
    def moving_mean(self, p_inputs, p_window):
        """
        To create a series of averages of different subsets of the full data set.
        """
        inputs  = np.array(p_inputs)
        outputs = np.zeros_like(inputs)
        if len(inputs.shape) == 1:
            outputs = np.convolve(inputs, np.ones((p_window,))/p_window, mode='same')
        else:
            for col in range(inputs.shape[1]):
                outputs[:,col] = np.convolve(inputs[:,col], np.ones((p_window,))/p_window, mode='same')
        return outputs


## -------------------------------------------------------------------------------------------------
    def save_plots(self, p_path, p_format, p_dpi_mul=1):
        """
        This method is used to save generated plots.

        Parameters:
            p_path          Path where file will be saved
            p_format        Format of the saved file.
                            Options: 'eps', 'jpg', 'png', 'pdf', 'svg'
            p_dpi_mul       Saving plots parameter

        Returns: 
            True, if plots where saved successfully. False otherwise.
        """

        num_plots = len(self.plots[0])
        if num_plots == 0: return False

        try:
            if not os.path.exists(p_path):
                os.makedirs(p_path)
            for idx in range(num_plots):
                self.plots[1][idx].savefig(p_path + os.sep + self.plots[0][idx] + "." + p_format, dpi=500*p_dpi_mul, bbox_inches = 'tight')
            return True
        except:
            return False