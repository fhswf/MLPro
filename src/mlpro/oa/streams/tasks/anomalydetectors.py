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
import matplotlib.patches as patches




## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class AnomalyDetector(OATask):
    """
    This is the base class for online anomaly detectors. It raises an event when an
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
        
        self.ano_id = 0
        self._anomalies      = {}
        self.visualize = p_visualize


## -------------------------------------------------------------------------------------------------
    def _run(self, p_inst_new: list, p_inst_del: list):
        pass


## -------------------------------------------------------------------------------------------------
    def get_anomalies(self):
        """
        This method returns the current list of anomalies. 

        Returns
        -------
        dict_of_anomalies : dict[AnomalyEvent]
            Current dictionary of anomalies.
        """

        return self._anomalies
    

## -------------------------------------------------------------------------------------------------
    def add_anomaly(self, p_anomaly):
        """
        Method to be used to add a new anomaly. Please use as part of your algorithm.

        Parameters
        ----------
        p_anomaly : Anomaly
            Anomaly object to be added.

        Returns
        -------
        p_anomaly : Anomaly
            Added Anomaly object.
        """

        self._anomalies[p_anomaly.get_id()] = p_anomaly
        return p_anomaly


## -------------------------------------------------------------------------------------------------
    def remove_anomaly(self, p_anomaly):
        """
        Method to remove an existing anomaly. Please use as part of your algorithm.

        Parameters
        ----------
        p_anomaly : Anomlay
            Anomlay object to be added.
        """
        #if p_anomaly.C_NAME == 'Point Anomaly':
        p_anomaly.remove_plot(p_refresh=True)
        del self._anomalies[p_anomaly.get_id()]


## -------------------------------------------------------------------------------------------------
    def raise_anomaly_event(self, p_instance : list[Instance]):

        self.ano_id +=1

        event = PointAnomaly(p_id=self.ano_id, p_instance=p_instance, p_visualize=self.visualize,
                             p_raising_object=self, p_det_time=str(p_instance[-1].get_tstamp()))

        event = self.add_anomaly(p_anomaly=event)

        if self.get_visualization(): 
            event.init_plot( p_figure=self._figure, p_plot_settings=self.get_plot_settings() )

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

        if not self.get_visualization(): return

        super().remove_plot(p_refresh=p_refresh)

        for anomaly in self._anomalies.values():
            anomaly.remove_plot(p_refresh=p_refresh)





## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class AnomalyDetectorExtended(AnomalyDetector):
    """
    This is the base class for multivariate online anomaly detectors. It raises an event when an
    anomaly is detected. This class has the added functionality to differentiate between different
    types of anomalies.

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
        
        self.group_anomalies = []
        self.group_anomalies_instances = []
        self._anomalies      = {}


## -------------------------------------------------------------------------------------------------
    def add_anomaly(self, p_anomaly):
        """
        Method to be used to add a new anomaly. Please use as part of your algorithm.

        Parameters
        ----------
        p_anomaly : Anomaly
            Anomaly object to be added.

        Returns
        -------
        p_anomaly : Anomaly
            Modified Anomaly object.
        """
        self.group_anomalies.append(p_anomaly)
        self.group_anomalies_instances.append(p_anomaly.get_instance()[-1])

        if len(self.group_anomalies_instances) > 1:

            if int(p_anomaly.get_instance()[0].get_id()) - 1 == int(self.group_anomalies_instances[-2].get_id()):

                if len(self.group_anomalies_instances) == 3:

                    for i in range(2):
                        self.remove_anomaly(self.group_anomalies[i])
                    self.ano_id -= 2
                    anomaly = GroupAnomaly(p_id=self.ano_id, p_instances=self.group_anomalies_instances, p_visualize=self.visualize,
                             p_raising_object=self, p_det_time=str(p_anomaly.get_instance()[-1].get_tstamp()))

                    self._anomalies[anomaly.get_id()] = anomaly
                    self.group_anomalies = []
                    self.group_anomalies.append(anomaly)
                    return anomaly

                elif len(self.group_anomalies_instances) > 3:
                    self.remove_anomaly(self.group_anomalies[0])
                    self.ano_id -= 1
                    anomaly = GroupAnomaly(p_id=self.ano_id, p_instances=self.group_anomalies_instances, p_visualize=self.visualize,
                             p_raising_object=self, p_det_time=str(p_anomaly.get_instance()[-1].get_tstamp()))
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
class AnomalyEvent (Event, Plottable):
    """
    This is the base class for anomaly events which can be raised by the anomaly detectors when an
    anomaly is detected.

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

    C_TYPE     = 'Event'
    C_NAME     = 'Anomaly'
    C_PLOT_ACTIVE           = True
    C_PLOT_STANDALONE       = False
    C_PLOT_VALID_VIEWS      = [ PlotSettings.C_VIEW_2D, 
                                PlotSettings.C_VIEW_3D, 
                                PlotSettings.C_VIEW_ND ]
    C_PLOT_DEFAULT_VIEW     = PlotSettings.C_VIEW_ND

## -------------------------------------------------------------------------------------------------
    def __init__(self,
                 p_id : int = None,
                 p_instance : Instance = None,
                 p_visualize : bool = False,
                 p_raising_object : object = None,
                 p_det_time : str = None,
                 **p_kwargs):
        super().__init__(p_raising_object=p_raising_object,
                         p_tstamp=p_det_time, **p_kwargs)
        Plottable.__init__( self, p_visualize = p_visualize )

        self.id = p_id
        self.instance = p_instance


## -------------------------------------------------------------------------------------------------
    def get_id(self):
        return self.id
    

## -------------------------------------------------------------------------------------------------
    def get_instance(self):
        return self.instance





## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class PointAnomaly (AnomalyEvent):
    """
    Event class for anomaly events when point anomalies are detected.
    
    """

    C_NAME      = 'Point Anomaly'

## -------------------------------------------------------------------------------------------------
    def __init__(self,
                 p_id : int = None,
                 p_instance : Instance = None,
                 p_visualize : bool = False,
                 p_raising_object : object = None,
                 p_det_time : str = None,
                 p_deviation : float=None,
                 **p_kwargs):
        super().__init__(p_id=p_id, p_instance=p_instance, p_visualize=p_visualize,
                         p_raising_object=p_raising_object, p_det_time=p_det_time,
                         **p_kwargs)
        
        self.id = p_id
        self.instance = p_instance


## -------------------------------------------------------------------------------------------------
    def _update_plot_nd(self, p_settings: PlotSettings, **p_kwargs):
        super()._update_plot_nd(p_settings, **p_kwargs)
    
        ylim  = p_settings.axes.get_ylim()
        label = self.C_NAME[0]
        self._plot_line1 = p_settings.axes.plot([self.get_instance()[-1].get_id(), self.get_instance()[-1].get_id()],
                                                ylim, color='r', linestyle='dashed', lw=1, label=label)[0]
        self._plot_line1_t1 = p_settings.axes.text(self.get_instance()[-1].get_id(), 0, label, color='r' )

    
## -------------------------------------------------------------------------------------------------
    def _remove_plot_nd(self):
        if self._plot_line1 is not None: self._plot_line1.remove()
        if self._plot_line1_t1 is not None: self._plot_line1_t1.remove()





## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class GroupAnomaly (AnomalyEvent):
    """
    Event class for anomaly events when group anomalies are detected.
    
    """

    C_NAME      = 'Group Anomaly'

## -------------------------------------------------------------------------------------------------
    def __init__(self,
                 p_id : int = None,
                 p_instances : Instance = None,
                 p_visualize : bool = False,
                 p_raising_object : object = None,
                 p_det_time : str = None,
                 p_mean : float=None,
                 p_mean_deviation : float=None,
                 **p_kwargs):
        super().__init__(p_id=p_id, p_instance=p_instances, p_visualize=p_visualize,
                         p_raising_object=p_raising_object, p_det_time=p_det_time,
                         **p_kwargs)
        
        self.id = p_id
        self.instance = p_instances
        

## -------------------------------------------------------------------------------------------------
    def _update_plot_nd(self, p_settings: PlotSettings, **p_kwargs):
        """
        Draw a shaded rectangular region on a plot.

        Parameters:
        ax (matplotlib.axes.Axes): The axes object to draw the shaded region on.
        x1, x2 (float): x-coordinates of the left and right edges of the rectangle.
        y1, y2 (float): y-coordinates of the bottom and top edges of the rectangle.
        color (str): Color of the shaded region.
        alpha (float): Transparency of the shaded region (default is 0.5).
        """
        super()._update_plot_nd(p_settings, **p_kwargs)
    
        label = self.C_NAME[0]
        x1 = self.get_instance()[0].get_id()
        x2 = self.get_instance()[-1].get_id()
        a=[]
        b=[]
        for instance in self.get_instance():
            a.append(instance.get_feature_data().get_values())
        for x in a:
            b.extend(x)
        y1 = min(b)
        y2 = max(b)

        rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=0, edgecolor='none', facecolor='yellow', alpha=0.01)
        self._plot_rectangle = p_settings.axes.add_patch(rect)
        self._plot_rectangle_t = p_settings.axes.text((x1+x2)/2, 0, label, color='b' )

    
## -------------------------------------------------------------------------------------------------
    def _remove_plot_nd(self):
        """
        Remove all shaded regions from a plot.

        """

        if self._plot_rectangle is not None: self._plot_rectangle .remove()
        if self._plot_rectangle_t is not None: self._plot_rectangle_t.remove()







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

