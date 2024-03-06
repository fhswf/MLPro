## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - A Synoptic Framework for Standardized Machine Learning Tasks
## -- Package : mlpro.oa.tasks.anomalydetectors
## -- Module  : anomalydetectors.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2023-06-08  0.0.0     SP       Creation
## -- 2023-09-12  1.0.0     SP       Release
## -- 2023-11-21  1.0.1     SP       Time Stamp update
## -- 2024-02-25  1.1.0     SP       Visualisation update
## -------------------------------------------------------------------------------------------------

"""
Ver. 1.1.0 (2023-02-25)
This module provides templates for anomaly detection to be used in the context of online adaptivity.
"""

from matplotlib.figure import Figure
from mlpro.bf.plot import PlotSettings
from mlpro.oa.streams.basics import *
from mlpro.oa.streams.basics import Instance, List
import numpy as np
from matplotlib.text import Text




## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class Anomaly (Id, Plottable):
    """
    Base class for an Anomaly. 

    Parameters
    ----------
    p_id
        Optional external id.
    p_visualize : bool
        Boolean switch for visualisation. Default = False.
    p_color : string
        Color of the anomaly during visualization.
    **p_kwargs
        Further optional keyword arguments.
    """

    C_PLOT_ACTIVE           = True
    C_PLOT_STANDALONE       = False
    C_PLOT_VALID_VIEWS      = [ PlotSettings.C_VIEW_2D, 
                                PlotSettings.C_VIEW_3D, 
                                PlotSettings.C_VIEW_ND ]
    C_PLOT_DEFAULT_VIEW     = PlotSettings.C_VIEW_ND

    C_CLUSTER_COLORS        = [ 'blue', 'orange', 'green', 'red', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan' ]

## -------------------------------------------------------------------------------------------------
    def __init__( self, 
                  p_id = None,
                  p_instance = None,
                  p_anomaly_type = None,
                  p_visualize : bool = False,
                  p_color = 'red',
                  **p_kwargs ):

        self._kwargs = p_kwargs.copy()
        Id.__init__( self, p_id = p_id )
        Plottable.__init__( self, p_visualize = p_visualize )

        self.id = p_id
        self.anomaly_type = p_anomaly_type
        self.instance = p_instance


## -------------------------------------------------------------------------------------------------
    def get_id(self):
        return self.id
    

## -------------------------------------------------------------------------------------------------
    def get_instance(self):
        return self.instance
    

## -------------------------------------------------------------------------------------------------
    def get_anomaly_type(self):
        return self.anomaly_type


## -------------------------------------------------------------------------------------------------
    def is_visualize(self):
        return self._visualize


## -------------------------------------------------------------------------------------------------
    def get_kwargs(self):
        return self._kwargs
    

## -------------------------------------------------------------------------------------------------
    def _init_plot_2d(self, p_figure: Figure, p_settings: PlotSettings):
        return super()._init_plot_2d(p_figure, p_settings)
    

## -------------------------------------------------------------------------------------------------
    def _init_plot_3d(self, p_figure: Figure, p_settings: PlotSettings):
        return super()._init_plot_3d(p_figure, p_settings)
    

## -------------------------------------------------------------------------------------------------
    def _init_plot_nd(self, p_figure: Figure, p_settings: PlotSettings):
        return super()._init_plot_nd(p_figure, p_settings)
    

## -------------------------------------------------------------------------------------------------
    def _update_plot_2d(self, p_settings: PlotSettings, **p_kwargs):
        return super()._update_plot_2d(p_settings, **p_kwargs)
    

## -------------------------------------------------------------------------------------------------
    def _update_plot_3d(self, p_settings: PlotSettings, **p_kwargs):
        return super()._update_plot_3d(p_settings, **p_kwargs)
    

## -------------------------------------------------------------------------------------------------
    def _update_plot_nd(self, p_settings: PlotSettings, **p_kwargs):

        super()._update_plot_nd(p_settings, **p_kwargs)

        ylim  = p_settings.axes.get_ylim()
        label = str(self.get_anomaly_type())[0]
        self._plot_line1 = p_settings.axes.plot([self.instance[-1].get_id(), self.instance[-1].get_id()],
                                                ylim, color='r', linestyle='dashed', lw=1, label=label)[0]
        self._plot_line1_t1 = p_settings.axes.text(self.instance[-1].get_id(), 0, label, color='r' )


## -------------------------------------------------------------------------------------------------
    def _remove_plot_2d(self):
        return super()._remove_plot_2d()
    

## -------------------------------------------------------------------------------------------------
    def _remove_plot_3d(self):
        return super()._remove_plot_3d()
    

## -------------------------------------------------------------------------------------------------
    def _remove_plot_nd(self):
        return super()._remove_plot_nd()





## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class AnomalyDetector(OATask):
    """
    This is the base class for multivariate online anomaly detectors. It raises an event when an
    anomaly is detected.

    """

    C_NAME          = 'Anomaly Detector'
    C_TYPE          = 'Anomaly Detector'

    C_PLOT_ACTIVE           = True
    C_PLOT_STANDALONE       = False

## -------------------------------------------------------------------------------------------------
    def __init__(self,
                 p_name:str = None,
                 p_range_max = StreamTask.C_RANGE_THREAD,
                 p_ada : bool = True,
                 p_duplicate_data : bool = False,
                 p_visualize : bool = False,
                 p_logging=Log.C_LOG_ALL,
                 **p_kwargs):

        super().__init__(p_name = p_name,
                         p_range_max = p_range_max,
                         p_ada = p_ada,
                         p_duplicate_data = p_duplicate_data,
                         p_visualize = p_visualize,
                         p_logging = p_logging,
                         **p_kwargs)
        
        self.ano_type = 'Anomaly'
        self.ano_id = 0
        self.group_anomalies = []
        self.group_anomalies_instances = []
        self._anomalies      = {}


## -------------------------------------------------------------------------------------------------
    def _run(self, p_inst_new: list, p_inst_del: list):
        pass


## -------------------------------------------------------------------------------------------------
    def get_anomalies(self) -> dict[Anomaly]:
        """
        This method returns the current list of anomalies. 

        Returns
        -------
        dict_of_anomalies : dict[Anomaly]
            Current dictionary of anomalies.
        """

        return self._anomalies
    

## -------------------------------------------------------------------------------------------------
    def add_anomaly(self, p_anomaly:Anomaly) -> bool:
        """
        Method to be used to add a new anomaly. Please use as part of your algorithm.

        Parameters
        ----------
        p_anomaly : Anomaly
            Anomaly object to be added.

        Returns
        -------
        successful : Bool
            True, if the anomaly has been added successfully. False otherwise.
        """
        self.group_anomalies.append(p_anomaly)
        self.group_anomalies_instances.append(p_anomaly.get_instance()[0])

        if len(self.group_anomalies_instances) > 1:

            if int(p_anomaly.get_instance()[0].get_id()) - 1 == int(self.group_anomalies_instances[-2].get_id()):

                if len(self.group_anomalies) == 3:

                    for i in range(2):
                        self.group_anomalies[i].remove_plot()
                        self.remove_anomaly(self.group_anomalies[i])
                    self.ano_id -= 2
                    anomaly = Anomaly(p_id=self.ano_id, p_instance=self.group_anomalies_instances, p_anomaly_type='Group Anomaly')
                    self._anomalies[anomaly.get_id()] = anomaly
                    self.group_anomalies = []
                    self.group_anomalies.append(anomaly)
                    return anomaly

                elif len(self.group_anomalies) > 3:
                    self.group_anomalies[0].remove_plot()
                    self.remove_anomaly(self.group_anomalies[0])
                    self.ano_id -= 1
                    anomaly = Anomaly(p_id=self.ano_id, p_instance=self.group_anomalies_instances, p_anomaly_type='Group Anomaly')
                    self._anomalies[anomaly.get_id()] = anomaly
                    self.group_anomalies = []
                    self.group_anomalies.append(anomaly)
                    return anomaly
                    
                else:
                    self._anomalies[p_anomaly.get_id()] = p_anomaly
                    return p_anomaly
            else:
                self.group_anomalies = []
                self.group_anomalies_instances = []
                self.group_anomalies.append(p_anomaly)
                self.group_anomalies_instances.append(p_anomaly.get_instance()[0])
                self._anomalies[p_anomaly.get_id()] = p_anomaly
                return p_anomaly
        else:
            self._anomalies[p_anomaly.get_id()] = p_anomaly
            return p_anomaly


## -------------------------------------------------------------------------------------------------
    def remove_anomaly(self, p_anomaly:Anomaly):
        """
        Method to remove an existing anomaly. Please use as part of your algorithm.

        Parameters
        ----------
        p_anomaly : Anomlay
            Anomlay object to be added.
        """

        p_anomaly.remove_plot(p_refresh=True)
        del self._anomalies[p_anomaly.get_id()]


## -------------------------------------------------------------------------------------------------
    def raise_anomaly_event(self, p_instance : Instance):

        #self.time_of_occ = p_instance.get_tstamp()
        #self.ano_type = self.get_anomaly_type(p_instance)
        #self.anomalies['ano_type'].append(self.ano_type)

        self.ano_id +=1

        anomaly = Anomaly(p_id=self.ano_id, p_instance=p_instance, p_anomaly_type='Point Anomaly')

        anomaly = self.add_anomaly(p_anomaly=anomaly)

        if self.get_visualization(): 
            anomaly.init_plot( p_figure=self._figure, p_plot_settings=self.get_plot_settings())

        if anomaly.get_anomaly_type() == 'Point Anomaly':
            event = PointAnomaly(p_raising_object=self, p_det_time=str(anomaly.get_instance()[-1].get_tstamp()),
                                 p_instance=p_instance)
        elif anomaly.get_anomaly_type() == 'Group Anomaly':
            event = GroupAnomaly(self, p_det_time=anomaly.get_instance()[-1].get_tstamp(),
                                 p_instances=self.group_anomalies)

        self._raise_event(event.C_NAME, event)

                 
## -------------------------------------------------------------------------------------------------
    def init_plot(self, p_figure: Figure = None, p_plot_settings: PlotSettings = None):

        if not self.get_visualization(): return

        super().init_plot( p_figure=p_figure, p_plot_settings=p_plot_settings)

        for anomaly in self._anomalies.values():
            anomaly.init_plot(p_figure=p_figure, p_plot_settings = p_plot_settings)
    

## -------------------------------------------------------------------------------------------------
    def update_plot(self, p_inst_new: List[Instance] = None, p_inst_del: List[Instance] = None, **p_kwargs):
    
        if not self.get_visualization(): return

        super().update_plot(p_inst_new, p_inst_del, **p_kwargs)

        for anomaly in self._anomalies.values():
            anomaly.update_plot(p_inst_new = p_inst_new, p_inst_del = p_inst_del, **p_kwargs)
    

## -------------------------------------------------------------------------------------------------
    def remove_plot(self, p_refresh: bool = True):

        self._anomalies.remove_plot(p_refresh=p_refresh)

        return super().remove_plot(p_refresh)

    



## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class AnomalyDetectorCB(AnomalyDetector):
    """
    This is the base class for cluster-based online anomaly detectors. It raises an event when an
    anomaly is detected in a cluster dataset.

    """

    C_TYPE = 'Cluster based Anomaly Detector'


## -------------------------------------------------------------------------------------------------
    def __init__(self,
                 p_threshold = 5.0,
                 p_centroid_threshold = 1.0,
                 p_name:str = None,
                 p_range_max = StreamTask.C_RANGE_THREAD,
                 p_ada : bool = True,
                 p_duplicate_data : bool = False,
                 p_visualize : bool = False,
                 p_logging=Log.C_LOG_ALL,
                 **p_kwargs):

        super().__init__(p_name = p_name,
                         p_range_max = p_range_max,
                         p_ada = p_ada,
                         p_duplicate_data = p_duplicate_data,
                         p_visualize = p_visualize,
                         p_logging = p_logging,
                         **p_kwargs)
        
        self.data_points = []
        self.anomaly_counter = 0
        self.anomaly_scores = []
        self.threshold = p_threshold
        self.centroid_thre = p_centroid_threshold
        self.centroids = []


## -------------------------------------------------------------------------------------------------
    def _run(self, p_inst_new: list, center: float, centroids: list):
        anomaly = None
        self.centroids.append(centroids)
        
        distance = np.linalg.norm(p_inst_new - center)
        if distance > self.threshold:
            anomaly = p_inst_new

        if len(centroids) > 10:
            self.centroids.pop(0)
        
        if len(self.centroids[-2]) != len(self.centroids[-1]):
            anomaly = p_inst_new
        differences = [abs(a - b) for a, b in zip(self.centroids[0], self.centroids[-1])]
        if any(difference >= self.centroid_thre for difference in differences):
            anomaly = p_inst_new

        if anomaly != None:
            self.anomaly_counter += 1
            event_obj = AnomalyEvent(p_raising_object=self, p_kwargs=self.data_points[-1]) 
            handler = self.event_handler
            self.register_event_handler(event_obj.C_NAME, handler)
            self._raise_event(event_obj.C_NAME, event_obj)
    




## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class AnomalyEvent (Event):
    """
    This is the base class for anomaly events which can be raised by the anomaly detectors when an
    anomaly is detected.

    """

    C_TYPE     = 'Event'
    C_NAME     = 'Anomaly'

## -------------------------------------------------------------------------------------------------
    def __init__(self, p_raising_object, p_det_time : str, **p_kwargs):
        super().__init__(p_raising_object=p_raising_object,
                         p_tstamp=p_det_time, **p_kwargs)





## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class PointAnomaly (AnomalyEvent):
    """
    Event class for anomaly events when point anomalies are detected.
    
    """

    C_NAME      = 'Point Anomaly'

## -------------------------------------------------------------------------------------------------
    def __init__(self, p_raising_object, p_det_time : str=None, p_instance : str=None,
                 p_deviation : float=None, **p_kwargs):
        super().__init__(p_raising_object=p_raising_object, p_det_time=p_det_time, **p_kwargs)





## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class GroupAnomaly (AnomalyEvent):
    """
    Event class for anomaly events when group anomalies are detected.
    
    """

    C_NAME      = 'Group Anomaly'

## -------------------------------------------------------------------------------------------------
    def __init__(self, p_raising_object, p_det_time : str, p_instances : list=None,
                 p_mean : float=None, p_mean_deviation : float=None, **p_kwargs):
        super().__init__(p_raising_object=p_raising_object,
                         p_det_time=p_det_time, **p_kwargs)




## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class ContextualAnomaly (AnomalyEvent):
    """
    Event class for anomaly events when contextual anomalies are detected
    
    """

    C_NAME      = 'Contextual Anomaly'

# -------------------------------------------------------------------------
    def __init__(self, p_raising_object, p_det_time :str, p_instances: str,  **p_kwargs):
        super().__init__(p_raising_object=p_raising_object,
                         p_det_time=p_det_time, **p_kwargs)





## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class DriftEvent (AnomalyEvent):
    """
    Event class to be raised when drift is detected.
    
    """

    C_NAME      = 'Drift'

## -------------------------------------------------------------------------------------------------
    def __init__(self, p_raising_object, p_det_time : str, p_magnitude : float, p_rate : float, **p_kwargs):
        super().__init__(p_raising_object=p_raising_object,
                         p_det_time=p_det_time, **p_kwargs)





## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class DriftEventCB (DriftEvent):
    """
    Event class to be raised when cluster drift is detected.
    
    """

    C_NAME      = 'Cluster based Drift'

## -------------------------------------------------------------------------------------------------
    def __init__(self, p_raising_object, p_det_time : str, **p_kwargs):
        super().__init__(p_raising_object=p_raising_object,
                         p_det_time=p_det_time, **p_kwargs)

