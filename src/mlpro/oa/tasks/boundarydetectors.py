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
## -------------------------------------------------------------------------------------------------

"""
Ver. 1.0.5 (2022-12-16)
This module provides pool of boundary detector object further used in the context of online adaptivity.
"""


from mlpro.bf.mt import Task as MLTask
import mlpro.bf.streams.tasks.windows as windows
from mlpro.oa import *
from typing import Union, Iterable, List






## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class BoundaryDetector(OATask):
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
    p_duplicate_data: bool
        True if duplicate of the original instances are to be processed. False by default.
    p_visualize: bool
        True to turn on the visualization.
    p_logging
        Logging level for the task, default is Log all.
    p_window: Window
        A predecessor window object, if any.
    p_scaler: float, np.ndarray
        A scaler vector to scale the detected boundaries.

    """

    C_NAME = 'Boundary Detector'



## -------------------------------------------------------------------------------------------------
    def __init__(self,
                 p_name:str = None,
                 p_range_max = MLTask.C_RANGE_THREAD,
                 p_ada : bool = True,
                 p_duplicate_data : bool = False,
                 p_visualize : bool = False,
                 p_logging=Log.C_LOG_ALL,
                 p_window: windows.Window = None,
                 p_scaler:Union[float, Iterable] = np.ones([1]),
                 **p_kwargs):


        super().__init__(p_name = p_name,
                         p_range_max = p_range_max,
                         p_ada = p_ada,
                         p_duplicate_data = p_duplicate_data,
                         p_visualize = p_visualize,
                         p_logging = p_logging
                         **p_kwargs)

        self._window = p_window
        self._scaler = p_scaler
        self._related_set: Set = None



## -------------------------------------------------------------------------------------------------
    def _adapt(self, p_inst_new:List[Instance], p_inst_del:List[Instance]):
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

        if p_inst_new is None: return

        adapted = False

        for inst in p_inst_new:
            if isinstance(inst, Instance):
                inst = inst.get_feature_data()

            # Storing the related set for events
            self._related_set = inst.get_related_set()

            dim = inst.get_related_set().get_dims()

            for i,value in enumerate(inst.get_values()):
                boundary = dim[i].get_boundaries()
                if len(boundary) == 0:
                    boundary = [0,0]
                    dim[i].set_boundaries(boundary)
                if value < boundary[0]:
                    dim[i].set_boundaries([value*self._scaler[i], boundary[i]])
                    adapted = True
                elif value > boundary[1]:
                    dim[i].set_boundaries([boundary[0],value*self._scaler[i]])
                    adapted = True
                else:
                    adapted = False or adapted

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
        if p_inst_new and not p_inst_del:
            self.adapt(p_inst_new=p_inst_new, p_inst_del=p_inst_del)


## -------------------------------------------------------------------------------------------------
    def _adapt_on_event(self, p_event_id:str, p_event_object:Event):
        """
        Event handler for Boundary Detector that adapts if the related event is raised
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
            boundaries = self._scaler*p_event_object.get_raising_object().get_boundaries()
            self._related_set = p_event_object.get_data()["p_set"]
            dims = [p_event_object.get_data()["p_set"].get_dim(i) for i in p_event_object.get_data()["p_set"].get_dim_ids()]
            for i,dim in enumerate(dims):
                if dim.get_boundaries()[0] != boundaries[i][0] or dim.get_boundaries()[1] != boundaries[i][1]:
                    dim.set_boundaries([boundaries[i]])
                    adapted = True
        except: raise ImplementationError("Event not raised by a window")
        return adapted


## -------------------------------------------------------------------------------------------------
    def get_related_set(self):

        return self._related_set


## -------------------------------------------------------------------------------------------------
    def _init_plot_2d(self, p_figure: Figure, p_settings: PlotSettings):
        """

        Parameters
        ----------
        p_figure
        p_settings

        Returns
        -------

        """
        pass


## -------------------------------------------------------------------------------------------------
    def _init_plot_3d(self, p_figure: Figure, p_settings: PlotSettings):
        """

        Parameters
        ----------
        p_figure
        p_settings

        Returns
        -------

        """
        pass


## -------------------------------------------------------------------------------------------------
    def _init_plot_nd(self, p_figure: Figure, p_settings: PlotSettings):
        """

        Parameters
        ----------
        p_figure
        p_settings

        Returns
        -------

        """
        pass