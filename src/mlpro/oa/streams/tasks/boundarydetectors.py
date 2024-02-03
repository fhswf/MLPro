## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - A Synoptic Framework for Standardized Machine Learning Tasks
## -- Package : mlpro.oa.tasks.boundarydetectors
## -- Module  : boundarydetectors.py
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
## -- 2023-11-19  1.2.4     DA       Bugfix in Method BoundaryDetection._adapt(): scaler management
## -------------------------------------------------------------------------------------------------

"""
Ver. 1.2.4 (2023-11-19)

This module provides pool of boundary detector object further used in the context of online adaptivity.
"""

import matplotlib.colors

from mlpro.bf.various import *
from mlpro.bf.plot import *
from mlpro.bf.math import *
from mlpro.bf.mt import Task as MLTask
from mlpro.oa.streams.basics import *
from typing import Union, Iterable, List



## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class BoundaryDetector (OATask):
    """
    This is the base class for Boundary Detector object. It raises event when a change in the current boundaries is
    detected based on the new data instances

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
    p_scaler: float, np.ndarray
        A scaler vector to scale the detected boundaries.

    """

    C_NAME                      = 'Boundary Detector'

    C_PLOT_ND_XLABEL_FEATURE    = 'Features'
    C_PLOT_ND_YLABEL            = 'Boundaries'

    C_PLOT_VALID_VIEWS          = [ PlotSettings.C_VIEW_ND ]

## -------------------------------------------------------------------------------------------------
    def __init__(self,
                 p_name:str = None,
                 p_range_max = MLTask.C_RANGE_THREAD,
                 p_ada : bool = True,
                 p_duplicate_data : bool = False,
                 p_visualize : bool = False,
                 p_logging=Log.C_LOG_ALL,
                 p_scaler:Union[float, Iterable] = np.ones(1),
                 **p_kwargs):

        super().__init__(p_name = p_name,
                         p_range_max = p_range_max,
                         p_ada = p_ada,
                         p_duplicate_data = p_duplicate_data,
                         p_visualize = p_visualize,
                         p_logging = p_logging,
                         **p_kwargs)

        self._scaler = p_scaler
        self._related_set: Set = None


## -------------------------------------------------------------------------------------------------
    def _adapt(self, p_inst_new:List[Instance]):
        """
        Method to check if the new instances exceed the current boundaries of the Set.

        Parameters
        ----------
        p_inst_new:list
            List of new instance/s added to the workflow

        Returns
        -------
        adapted : bool
            Returns true if there is a change of boundaries, false otherwise.
        """

        adapted = False

        for inst in p_inst_new:
            if isinstance(inst, Instance):
                feature_data = inst.get_feature_data()
            else:
                feature_data = inst

            # Storing the related set for events
            self._related_set = feature_data.get_related_set()

            dim = feature_data.get_related_set().get_dims()

            if len(self._scaler) == 1:
                self._scaler = np.repeat(self._scaler[0], len(dim), axis=0)

            for i,value in enumerate(feature_data.get_values()):
                boundary = dim[i].get_boundaries()
                if len(boundary) == 0 or boundary is None:
                    boundary = [ 0,0 ]
                    dim[i].set_boundaries(boundary)
                    adapted = True

                if value < boundary[0]:
                    dim[i].set_boundaries([value*self._scaler[i], boundary[1]])
                    adapted = True
                elif value > boundary[1]:
                    dim[i].set_boundaries([boundary[0],value*self._scaler[i]])
                    adapted = True

        return adapted


## -------------------------------------------------------------------------------------------------
    def _run(self, p_inst_new:List[Element], p_inst_del:List[Element]):
        """
        Method to run the boundary detector task

        Parameters
        ----------
            p_inst_new:list
                List of new instance/s added to the workflow
            p_inst_del:list
                List of old obsolete instance/s removed from the workflow
        """

        if p_inst_new:
            self.adapt(p_inst_new=p_inst_new, p_inst_del=p_inst_del)


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
            bd_new = self._scaler*p_event_object.get_raising_object().get_boundaries()
            self._related_set = p_event_object.get_data()["p_related_set"]
            dims = self._related_set.get_dims()

            for i,dim in enumerate(dims):
                bd_dim_current = dim.get_boundaries()
                bd_dim_new     = bd_new[i]

                if ( bd_dim_new[0] != bd_dim_current[0] ) or ( bd_dim_new[1] != bd_dim_current[1] ):
                    dim.set_boundaries(bd_dim_new)
                    adapted = True
        except: 
            raise ImplementationError("Event not raised by a window")
        
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

        if not p_settings.axes:
            self.axes = p_figure.add_axes([0.1,0.1,0.7,0.8])
            self.axes.set_xlabel(self.C_PLOT_ND_XLABEL_FEATURE)
            self.axes.set_ylabel(self.C_PLOT_ND_YLABEL)
            self.axes.grid(visible=True)
            p_settings.axes = self.axes

        else:
            self.axes = p_settings.axes

        self._plot_nd_plots = None


## --------------------------------------------------------------------------------------------------
    def _update_plot_nd( self,
                         p_settings : PlotSettings,
                         p_inst_new : list,
                         p_inst_del : list,
                         **p_kwargs ):
        """
        Default N-dimensional plotting implementation for Boundary Detector tasks. See class mlpro.bf.plot.Plottable
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

        if ((p_inst_new is None) or (len(p_inst_new) == 0)
                and ((p_inst_del is None) or len(p_inst_del) ==0)) : return

        dims = self.get_related_set().get_dims()

        if self._plot_nd_plots is None:

            self._plot_nd_plots = {}
            heights = list(repeat(0, len(dims)))
            bottoms = list(repeat(0, len(dims)))
            labels = [i.get_name_long() for i in self.get_related_set().get_dims()]
            bars = self.axes.bar(range(len(dims)), height=heights, bottom=bottoms,
                color = matplotlib.colors.XKCD_COLORS)
            for i,(dim,bar) in enumerate(zip(dims, bars)):
                self._plot_nd_plots[dim] = bar
                self._plot_nd_plots[dim].set_label(str(i)+'. '+dim.get_name_long())
            self.axes.set_xticks(range(len(labels)))
            self.axes.legend(loc='center left', bbox_to_anchor=(1, 0.5))


        for dim in self._plot_nd_plots.keys():
            upper_boundary = dim.get_boundaries()[1]
            lower_boundary = dim.get_boundaries()[0]
            self._plot_nd_plots[dim].set_y(lower_boundary)
            self._plot_nd_plots[dim].set_height(upper_boundary-lower_boundary)


            # Setting the plot limits
            ylim = self.axes.get_ylim()

            if (ylim[0] > lower_boundary) or (ylim[1] < upper_boundary):
                if lower_boundary >= 0:
                    plot_boundary = [0, upper_boundary]
                else:
                    plot_boundary = [-max(upper_boundary, -(lower_boundary)), max(upper_boundary, -(lower_boundary))]
                self.axes.set_ylim(plot_boundary)

