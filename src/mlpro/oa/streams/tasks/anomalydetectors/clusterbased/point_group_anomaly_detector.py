## -- ----------------------------------------------------------------------------------------------
## -- Project : MLPro - The integrative middleware framework for standardized machine learning
## -- Package : mlpro.oa.tasks.anomalydetectors.cb_detectors
## -- Module  : point_group_anomaly_detector.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2025-03-28  0.0.0     DS       Creation
## -- 2025-04-03  0.0.1     DS/DA    Buffering of anomalies added.
## -- 2025-04-18  0.0.2     DS       Refactoring - New seperate classes for spatial and temporal group anomalies.  
## -- 2025-04-29  0.1.0     DS/DA    Refactoring
## -- 2025-05-04  0.1.1     DS       Design extensions
## -------------------------------------------------------------------------------------------------

"""
Ver. 0.1.1 (2025-05-04)

This module provides cluster based point and group anomaly detector algorithm.
"""

from mlpro.bf.various import Log
from mlpro.bf.math.properties import *
from mlpro.bf.streams import InstDict, InstTypeNew, Stream
from mlpro.oa.streams import OAStreamTask
from mlpro.oa.streams.tasks.clusteranalyzers import ClusterAnalyzer, Cluster
from mlpro.oa.streams.tasks.clusteranalyzers.clusters.properties import cprop_size, cprop_size_prev
from mlpro.oa.streams.tasks.anomalydetectors.clusterbased.basics import AnomalyDetectorCB, AnomalyDetectorCBSingle, AnomalyDetectorCBMulti
from mlpro.oa.streams.tasks.anomalydetectors.anomalies.clusterbased import AnomalyCB, PointAnomaly, GroupAnomaly, SpatialGroupAnomaly, TemporalGroupAnomaly


## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class AnomalyDetectorCBPA(AnomalyDetectorCBSingle):
    """
    Implementation of a cluster-based detector for point and group anomalies.

    Parameters
    ----------
    ...
    p_clusterer : ClusterAnalyzer
        The cluster analyzer to be used for clustering the data stream.
    p_property : PropertyDefinition         
        The property to be observed for anomalies.
    p_cls_point_anomaly : type
        The class of the point anomaly to be created.
    p_cls_spatial_group_anomaly : type
        The class of the spatial group anomaly to be created.
    p_cls_temporal_group_anomaly : type
        The class of the temporal group anomaly to be created.
    p_name : str
        The name of the anomaly detector.
    p_range_max : OAStreamTask.C_RANGE_THREAD
        The maximum range of the data stream.
    p_ada : bool
        Whether to use adaptive data analysis.
    p_duplicate_data : bool
        Whether to allow duplicate data in the data stream.
    p_visualize : bool
        Whether to visualize the anomalies.
    p_logging : Log.C_LOG_ALL
        The logging level for the anomaly detector.
    p_anomaly_buffer_size : int = 100
        The size of the anomaly buffer.
    ...
    """

    C_NAME = 'Point anomaly'
    C_PROPERTY_DEFINITIONS : PropertyDefinition = []
    
## -------------------------------------------------------------------------------------------------
    def __init__( self,
                  p_clusterer : ClusterAnalyzer,
                  p_property : PropertyDefinition = cprop_size_prev,
                  p_cls_point_anomaly : type = PointAnomaly,
                  p_name : str = None,
                  p_range_max = OAStreamTask.C_RANGE_THREAD,
                  p_ada : bool = True,
                  p_duplicate_data : bool = False,
                  p_visualize : bool = False,
                  p_logging=Log.C_LOG_ALL,
                  p_anomaly_buffer_size : int = 100,
                  **p_kwargs ):
        
        self._property = p_property
        self._cls_point_anomaly = p_cls_point_anomaly
        
        super().__init__( p_clusterer=p_clusterer,
                          p_name = p_name,
                          p_range_max = p_range_max,
                          p_ada = p_ada,
                          p_duplicate_data = p_duplicate_data,
                          p_visualize = p_visualize,
                          p_logging= p_logging,                        
                          **p_kwargs)


## -------------------------------------------------------------------------------------------------    
    def _run_algorithm( self, 
                        p_inst: InstDict) :
        """
        custom methid for detectiong cluster-based point anomalies.
        """

        # 1 Get all the clusters from the clusterer
        clusters = self._clusterer.clusters

        for cluster in clusters.values():

        # 2 Get the cluster property to be observed
            prop_cluster_size : Property = getattr(cluster, p_property)

            # 2.1 Check for the  spatial group anomalies
            if (prop_cluster_size.value == 1) and (prop_cluster_size.value_prev is None):
            
                # 2.1.1 Create a new point anomaly
                point_anomaly = self._cls_point_anomaly( p_clusters = {p_cluster.id : p_cluster},
                                                     p_tstamp = self.get_tstamp(),
                                                     p_visualize = self.get_visualize,
                                                     p_raising_object = self)

                self._raise_anomaly_event(p_anomaly = point_anomaly)
        

## ----------------------------------------------------------------------------------
    def _triage_anomaly( self, p_anomaly : AnomalyCB ):
        """
        Custom method for anomaly triage for point anomalies.
        This method checks if the point anomaly is still valid based on the cluster size.
        """
        # 1 Iterate over all anomalies
        for anomaly_id, anomaly in self.cb_anomalies.items():

            if isinstance(anomaly, self._cls_point_anomaly):
                #1.1 Get the single cluster associated with the anomaly
                cluster_id, cluster = next(iter(anomaly.clusters.items()))

                #1.2 Get the cluster property to be observed
                prop_cluster_size : Property = getattr(cluster, cprop_size_prev)

                # 1.3 Check if the anomaly is still valid, if not remove it
                if prop_cluster_size.value != 1:
                    self._remove_anomaly( p_anomaly = anomaly )





## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class AnomalyDetectorCBSGA(AnomalyDetectorCBPA):
    """
    Implementation of a cluster-based detector for spatial group anomalies.

    Parameters
    ----------          
    ...
    p_clusterer : ClusterAnalyzer
        The cluster analyzer to be used for clustering the data stream.
    p_property : PropertyDefinition
        The property to be observed for anomalies.
    p_cls_spatial_group_anomaly : type
        The class of the spatial group anomaly to be created.
    p_cls_point_anomaly : type
        The class of the point anomaly to be created.
    p_name : str
        The name of the anomaly detector.
    p_range_max : OAStreamTask.C_RANGE_THREAD
        The maximum range of the data stream.
    p_ada : bool    
        Whether to use adaptive data analysis.
    p_duplicate_data : bool     
        Whether to allow duplicate data in the data stream.
    p_visualize : bool
        Whether to visualize the anomalies.
    p_logging : Log.C_LOG_ALL
        The logging level for the anomaly detector.
    p_anomaly_buffer_size : int = 100
        The size of the anomaly buffer.
    p_thres_percent : float = 0.05
        The threshold percentage for the cluster size.
    """

    C_NAME = 'Spatial group anomaly'
    C_PROPERTY_DEFINITIONS : PropertyDefinition = []

## -------------------------------------------------------------------------------------------------
    def __init__( self,
                  p_clusterer : ClusterAnalyzer,
                  p_property : PropertyDefinition,
                  p_cls_spatial_group_anomaly : type = SpatialGroupAnomaly,
                  p_name : str = None,
                  p_range_max = OAStreamTask.C_RANGE_THREAD,
                  p_ada : bool = True,
                  p_duplicate_data : bool = False,
                  p_visualize : bool = False,
                  p_logging=Log.C_LOG_ALL,
                  p_anomaly_buffer_size : int = 100,
                  p_thres_percent : float = 0.05,
                  **p_kwargs ):

        
        self._cls_spatial_group_anomaly = p_cls_spatial_group_anomaly
        self._group_anomaly_det = p_group_anomaly_det
        self._thres_percent = p_thres_percent
        self.clusters_to_remove = []
        

        super().__init__( p_clusterer=p_clusterer,
                          p_name = p_name,
                          p_range_max = p_range_max,
                          p_ada = p_ada,
                          p_duplicate_data = p_duplicate_data,
                          p_visualize = p_visualize,
                          p_logging= p_logging,                        
                          **p_kwargs)


## -------------------------------------------------------------------------------------------------
    def _run_algorithm( self,
                        p_inst: InstDict):

        super()._run_algorithm( p_inst = p_inst )


        # 1 Get all the clusters from the clusterer
        clusters = self._clusterer.clusters

        # 3 Get average cluster size
        avg_cluster_size = sum([getattr(c, cprop_size[0]).value for c in clusters]) / len(clusters)

        # 4 Calculater the threshold for the cluster size
        self._thres_size = avg_cluster_size * self._thres_percent 


        for cluster in clusters.values():

            # 4.1 Get the cluster property to be observed
            prop_cluster_size : Property = getattr(cluster, p_property)


            # 4.2 Check for the  spatial group anomalies
            if 1 < prop_cluster_size.value <= self._thres_size:

                # 4.2.1 Create a new spatial group anomaly 
                try:
                    cb_anomaly = self.cb_anomalies[cluster.id]

                    if  type(cb_anomaly) == self._cls_point_anomaly:
                        self._remove_anomaly(p_anomaly = cb_anomaly)
                        create_anomaly = True
                    else:
                        create_anomaly = False
                        
                except:

                    create_anomaly = True

                # 4.2.2 Create a new spatial group anomaly    
                if create_anomaly:
                    spatial_group_anomaly = self._cls_spatial_group_anomaly( p_clusters = {cluster.id : cluster},
                                                                            p_tstamp = self.get_tstamp(),
                                                                            p_visualize = self.get_visualize,
                                                                            p_raising_object = self)

                    # 4.2.2.1 Raise an anomaly event
                    self._raise_anomaly_event( p_anomaly = spatial_group_anomaly)


## -------------------------------------------------------------------------------------------------
    def _triage_anomaly( self, p_anomaly : AnomalyCB ):
        """
        Custom method for anomaly triage.
        """
        # 1 Iterate over all anomalies
        for anomaly_id, anomaly in self.cb_anomalies.items():

            if isinstance(anomaly, self._cls_spatial_group_anomaly):
                #1.1 Get the single cluster associated with the anomaly
                cluster_id, cluster = next(iter(anomaly.clusters.items()))

                #1.2 Get the cluster property to be observed
                prop_cluster_size : Property = getattr(cluster, cprop_size_prev)
                
                # 1.3 Check if the anomaly is still valid, if not remove it
                if not (1 < prop_cluster_size.value <= self._thres_size):
                    self._remove_anomaly( p_anomaly = anomaly )





## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class AnomalyDetectorCBTGA(AnomalyDetectorCBSGA):
    """
    Implementation of a cluster-based detector for temporal group anomalies.

    parameters
    ----------
    ...
    p_clusterer : ClusterAnalyzer
        The cluster analyzer to be used for clustering the data stream.
    p_property : PropertyDefinition
        The property to be observed for anomalies.
    p_cls_temporal_group_anomaly : type
        The class of the temporal group anomaly to be created.
    p_cls_point_anomaly : type
        The class of the point anomaly to be created.
    p_name : str
        The name of the anomaly detector.
    p_range_max : OAStreamTask.C_RANGE_THREAD
        The maximum range of the data stream.
    p_ada : bool
        Whether to use adaptive data analysis.
    p_duplicate_data : bool
        Whether to allow duplicate data in the data stream.
    p_visualize : bool
        Whether to visualize the anomalies.
    p_logging : Log.C_LOG_ALL
        The logging level for the anomaly detector.
    p_anomaly_buffer_size : int = 100
        The size of the anomaly buffer.
    p_thres_percent : float = 0.05
        The threshold percentage for the cluster size.
    p_thres_temporal : int = 2
        The threshold for the number of temporal anomalies.
    """

    C_NAME = 'Temporal group anomaly'
    C_PROPERTY_DEFINITIONS : PropertyDefinition = []

## -------------------------------------------------------------------------------------------------
    def __init__( self,
                  p_clusterer : ClusterAnalyzer,
                  p_property : PropertyDefinition,
                  p_cls_temporal_group_anomaly : type = TemporalGroupAnomaly,
                  p_cls_point_anomaly : type = PointAnomaly,
                  p_name : str = None,
                  p_range_max = OAStreamTask.C_RANGE_THREAD,
                  p_ada : bool = True,
                  p_duplicate_data : bool = False,
                  p_visualize : bool = False,
                  p_logging=Log.C_LOG_ALL,
                  p_anomaly_buffer_size : int = 100,
                  p_thres_percent : float = 0.05,
                  p_thres_temporal : int = 2,
                  **p_kwargs ):
        
        self._cls_temporal_group_anomaly = p_cls_temporal_group_anomaly
        self._cls_point_anomaly = p_cls_point_anomaly
        self._cb_anomalies = {}
        self._thres_percent = p_thres_percent
        self._thres_temporal = p_thres_temporal
        self._temporal_anomalies = {}

        super().__init__( p_clusterer=p_clusterer,
                          p_name = p_name,
                          p_range_max = p_range_max,
                          p_ada = p_ada,
                          p_duplicate_data = p_duplicate_data,
                          p_visualize = p_visualize,
                          p_logging= p_logging,                        
                          **p_kwargs)


## -------------------------------------------------------------------------------------------------
    def _run_algorithm( self,
                        p_inst: InstDict):

        super()._run_algorithm( p_inst = p_inst )


        # 1 Get all the clusters from the clusterer
        clusters = self._clusterer.clusters

        # 2 Check for the  spatial group anomalies
        if len(self._temporal_anomalies[point_anomaly]) >= self._thres_temporal:
            
            try:
                cb_anomaly = self._cb_anomalies[p_cluster.id]

                create_anomaly = (type(cb_anomaly) == self._cls_point_anomaly or type(cb_anomaly) == self._cls_temporal_group_anomaly)

            except:
                create_anomaly = True

            # 2.1 Create a new temporal group anomaly 
            if create_anomaly:
                temporal_anomaly = self._cls_temporal_group_anomaly( p_clusters = {p_cluster.id : p_cluster},
                                                                     p_tstamp = self.get_tstamp(),
                                                                     p_visualize = self.get_visualize,
                                                                     p_raising_object = self)
                # 2.1.1 Raise an anomaly event
                self._raise_anomaly_event(p_anomaly = temporal_anomaly)


## -------------------------------------------------------------------------------------------------
    def _triage_anomaly( self, p_anomaly : AnomalyCB ):
        """
        Custom method for anomaly triage.
        """
        # 1 Iterate over all anomalies
        for anomaly_id, anomaly in self.cb_anomalies.items():

            if isinstance(anomaly, self._cls_temporal_group_anomaly):
                #1.1 Get the single cluster associated with the anomaly
                cluster_id, cluster = next(iter(anomaly.clusters.items()))
                # 1.2 Check if the anomaly is still valid
                if len(self._temporal_anomalies[cluster_id]) < self._thres_temporal:
                    # 1.2.1 If anomaly no longer exists, remove anomaly
                    self._remove_anomaly( p_anomaly = anomaly )