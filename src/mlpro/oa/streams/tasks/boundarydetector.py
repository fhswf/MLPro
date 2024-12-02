## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - The integrative middleware framework for standardized machine learning
## -- Package : mlpro.oa.streams.tasks
## -- Module  : boundarydetector.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2022-10-17  0.0.0     LSB      Creation
## -- 2022-10-21  1.0.0     LSB      Release
## -- 2022-12-07  1.0.1     LSB      Refactoring for scaler factor
## -- 2022-12-08  1.0.2     LSB      Compatibiltty for instance and Element objects
## -- 2022-12-12  1.0.3     DA       Corrected signature of method _adapt_on_event()
## -- 2022-12-13  1.0.4     LSB      Refactoring
## -- 2022-12-16  1.0.5     LSB      Refactoring for get_related_set method
## -- 2022-12-20  1.0.6     LSB      Bug Fixes
## -- 2022-12-20  1.1.0     LSB      ND Visualization
## -- 2022-12-20  1.1.1     LSB      Refactoring visualization
## -- 2022-12-20  1.1.2     LSB      Bug Fix
## -- 2022-12-28  1.1.3     DA       Class BoundaryDetector: 
## --                                - added constant C_PLOT_VALID_VIEWS
## --                                - removed methods init_plot_2d/3d, update_plot_2d/3d
## -- 2022-12-30  1.1.4     DA       Removed the plot title
## -- 2023-02-02  1.1.5     DA       Method BoundaryDetector._init_plot_2D: removed figure creation
## -- 2023-02-13  1.1.6     SY       Bug Fix: Solving issue, when the first data is lower than 1
## -- 2023-02-13  1.1.7     LSB      Bug Fix: Setting initial boundaries to [0,0]
## -- 2023-04-09  1.2.0     DA       Refactoring of method BoundaryDetector._adapt()
## -- 2023-05-02  1.2.1     DA       Class BoundaryDetector
## --                                - constructor: removed parameter p_window
## --                                - method _adapt(): removed unnecessary code
## -- 2023-05-20  1.2.2     DA       Method BoundaryDetector._adapt_on_event: refactoring, corrections
## -- 2023-05-21  1.2.3     LSB      Bug Fix : p_scaler shall be generated as a vertical array
## -- 2023-11-19  1.2.4     DA       Bugfix in method BoundaryDetector._adapt(): scaler management
## -- 2024-05-12  1.3.0     DA       Removed the scaler functionality from BoundaryDetector
## -- 2024-05-22  1.4.0     DA       Refactoring and correction in BoundaryDetector._adapt()
## -- 2024-10-29  1.5.0     DA       - Refactoring
## --                                - Pseudo-implementation of BoundaryDetector._adapt_reverse()
## -- 2024-11-05  1.5.1     DA       Bugfix in method BoundaryDetector._upate_plot_nd()
## -------------------------------------------------------------------------------------------------

"""
Ver. 1.5.1 (2024-11-05)

This module provides pool of boundary detector object further used in the context of online adaptivity.

"""

from itertools import repeat

import matplotlib.colors
from matplotlib.figure import Figure

from mlpro.bf.various import Log
from mlpro.bf.exceptions import ImplementationError
from mlpro.bf.plot import PlotSettings
from mlpro.bf.mt import Task
from mlpro.bf.events import Event
from mlpro.bf.math import Set
from mlpro.bf.streams import Instance, InstDict
from mlpro.oa.streams.basics import OAStreamTask




## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class BoundaryDetector (OAStreamTask):
    """
    This class provides the functionality of boundary observation of incoming instances. It raises 
    event C_EVENT_ADAPTED when a change in the current boundaries is detected.

    Parameters
    ----------
    p_name: str, Optional.
        Name of the task.
    p_range_max
        Processing range of the task. Default is thread.
    p_ada: bool
        True if the task has adaptivity. Default is True.
    p_duplicate_data : bool
        If True, instances will be duplicated before processing. Default = False.
    p_visualize: bool
        True to turn on the visualization.
    p_logging
        Logging level for the task, default is Log all.

    """

    C_NAME                      = 'Boundary Detector'

    C_PLOT_ND_XLABEL_FEATURE    = 'Features'
    C_PLOT_ND_YLABEL            = 'Boundaries'

    C_PLOT_STANDALONE           = True
    C_PLOT_VALID_VIEWS          = [ PlotSettings.C_VIEW_ND ]
    C_PLOT_DEFAULT_VIEW         = PlotSettings.C_VIEW_ND

## -------------------------------------------------------------------------------------------------
    def __init__( self,
                  p_name:str = None,
                  p_range_max = Task.C_RANGE_THREAD,
                  p_ada : bool = True,
                  p_duplicate_data : bool = False,
                  p_visualize : bool = False,
                  p_logging=Log.C_LOG_ALL,
                  **p_kwargs ):

        super().__init__( p_name = p_name,
                          p_range_max = p_range_max,
                          p_ada = p_ada,
                          p_duplicate_data = p_duplicate_data,
                          p_visualize = p_visualize,
                          p_logging = p_logging,
                          **p_kwargs )

        self._related_set: Set = None


## -------------------------------------------------------------------------------------------------
    def _adapt(self, p_inst_new: Instance) -> bool:
        """
        Method to check if the new instances exceed the current boundaries of the Set.

        Parameters
        ----------
        p_inst_new : Instance
            New instance/s added to the workflow

        Returns
        -------
        adapted : bool
            Returns true if there is a change of boundaries, false otherwise.
        """

        adapted = False

        feature_data = p_inst_new.get_feature_data()

        # Storing the related set for events
        self._related_set = feature_data.get_related_set()

        dim = feature_data.get_related_set().get_dims()

        for i,value in enumerate(feature_data.get_values()):
            boundary = dim[i].get_boundaries()
            if len(boundary) == 0 or boundary is None:
                dim[i].set_boundaries([value,value])
                adapted = True
                continue

            if value < boundary[0]:
                dim[i].set_boundaries([value, boundary[1]])
                adapted = True
            elif value > boundary[1]:
                dim[i].set_boundaries([boundary[0],value])
                adapted = True

        return adapted
    
    
## -------------------------------------------------------------------------------------------------
    def _adapt_reverse(self, p_inst_del: Instance):
        """
        Pseudo-implementation
        """
        return False


## -------------------------------------------------------------------------------------------------
    def _run(self, p_inst: InstDict):
        """
        Method to run the boundary detector task

        Parameters
        ----------
        p_inst : InstDict
            Instances to be processed.
        """

        self.adapt(p_inst=p_inst)


## -------------------------------------------------------------------------------------------------
    def _adapt_on_event(self, p_event_id:str, p_event_object:Event):
        """
        Event handler for Boundary Detector that adapts if the related event is raised.

        Parameters
        ----------
            p_event_id
                The event id related to the adaptation.
            p_event_obj:Event
                The event object related to the raised event.

        Returns
        -------
            bool
                Returns true if adapted, false otherwise.
        """

        adapted = False
        
        try:
            bd_new = p_event_object.get_raising_object().get_boundaries()
        except: 
            raise ImplementationError("Event not raised by a window")
        
        self._related_set = p_event_object.get_data()["p_related_set"]
        dims = self._related_set.get_dims()

        for i,dim in enumerate(dims):
            bd_dim_current = dim.get_boundaries()
            bd_dim_new     = bd_new[i]

            if ( bd_dim_new[0] != bd_dim_current[0] ) or ( bd_dim_new[1] != bd_dim_current[1] ):
                dim.set_boundaries(bd_dim_new)
                adapted = True
        
        return adapted


## -------------------------------------------------------------------------------------------------
    def get_related_set(self):
        return self._related_set


## -------------------------------------------------------------------------------------------------
    def _init_plot_nd(self, p_figure: Figure, p_settings: PlotSettings):
        """
        Custom method to initialize plot for Boundary Detectors tasks for N-dimensional plotting.

        Parameters
        ----------
        p_figure:Figure
            Figure to host the plot
        p_settings: PlotSettings
            PlotSettings objects with specific settings for the plot

        """

        if p_settings.axes is None:
            p_settings.axes = p_figure.add_axes([0.1,0.1,0.7,0.8])
            p_settings.axes.set_xlabel(self.C_PLOT_ND_XLABEL_FEATURE)
            p_settings.axes.set_ylabel(self.C_PLOT_ND_YLABEL)
            p_settings.axes.grid(visible=True)

        self._plot_nd_plots = None


## --------------------------------------------------------------------------------------------------
    def _update_plot_nd( self,
                         p_settings : PlotSettings,
                         p_inst : InstDict,
                         **p_kwargs ):
        """
        Default N-dimensional plotting implementation for Boundary Detector tasks. See class mlpro.bf.plot.Plottable
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

        # 0 Intro
        if len(p_inst) == 0: return

        dims = self.get_related_set().get_dims()


        # 1 Set up plotting 
        if self._plot_nd_plots is None:

            self._plot_nd_plots = {}
            heights = list(repeat(0, len(dims)))
            bottoms = list(repeat(0, len(dims)))
            labels = [i.get_name_long() for i in self.get_related_set().get_dims()]
            bars = p_settings.axes.bar(range(len(dims)), height=heights, bottom=bottoms,
                color = matplotlib.colors.XKCD_COLORS)
            for i,(dim,bar) in enumerate(zip(dims, bars)):
                self._plot_nd_plots[dim] = bar
                self._plot_nd_plots[dim].set_label(str(i)+'. '+dim.get_name_long())
            p_settings.axes.set_xticks(range(len(labels)))
            p_settings.axes.legend(loc='center left', bbox_to_anchor=(1, 0.5))


        # 2 Update boundary bars and ax limits
        plot_boundary = [None,None]

        for dim in self._plot_nd_plots.keys():
            upper_boundary = dim.get_boundaries()[1]
            lower_boundary = dim.get_boundaries()[0]
            self._plot_nd_plots[dim].set_y(lower_boundary)
            self._plot_nd_plots[dim].set_height(upper_boundary-lower_boundary)

            if ( plot_boundary[0] is None ) or ( plot_boundary[0] > lower_boundary):
                plot_boundary[0] = lower_boundary

            if ( plot_boundary[1] is None ) or ( plot_boundary[1] < upper_boundary):
                plot_boundary[1] = upper_boundary

        p_settings.axes.set_ylim(plot_boundary)

