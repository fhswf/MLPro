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
## -- 2025-05-06  0.2.0     DS/DA    Refactoring
## -- 2025-05-11  0.2.1     DS       Design extensions
## -- 2025-05-19  0.2.2     DS       Bug fixes
## -- 2025-05-27  0.3.0     DS       Design extensions
## -------------------------------------------------------------------------------------------------

"""
Ver. 0.3.0 (2025-05-27)

This module provides cluster based point and group anomaly detector algorithm.
"""

from mlpro.bf.various import Log
from mlpro.bf.math.properties import Property, PropertyDefinition
from mlpro.bf.streams import InstDict, InstTypeNew, Stream, Instance
from mlpro.oa.streams import OAStreamTask
from mlpro.oa.streams.tasks.clusteranalyzers import ClusterAnalyzer, Cluster
from mlpro.oa.streams.tasks.clusteranalyzers.clusters.properties import cprop_size, cprop_size_prev
from mlpro.oa.streams.tasks.changedetectors.anomalydetectors.clusterbased.basics import AnomalyDetectorCB
from mlpro.oa.streams.tasks.changedetectors.anomalydetectors.anomalies.clusterbased import AnomalyCB, PointAnomaly, GroupAnomaly, SpatialGroupAnomaly, TemporalGroupAnomaly


## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class AnomalyDetectorCBPA(AnomalyDetectorCB):
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
        self._latest_anomaly = None
        
        super().__init__( p_clusterer=p_clusterer,
                          p_name = p_name,
                          p_range_max = p_range_max,
                          p_ada = p_ada,
                          p_duplicate_data = p_duplicate_data,
                          p_visualize = p_visualize,
                          p_logging= p_logging,                        
                          **p_kwargs)


## -------------------------------------------------------------------------------------------------    
    def _detect_cb_anomalies( self, 
                              p_inst: Instance ): 
        """
        custom methid for detectiong cluster-based point anomalies.
        """

        # 1 Get all the clusters from the clusterer
        clusters = self._clusterer.clusters

        for cluster in clusters.values():

        # 2 Get the cluster property to be observed
            prop_cluster_size : Property = getattr(cluster, self._property)

            # 2.1 Check for the  spatial group anomalies
            if (prop_cluster_size.value == 1) and (prop_cluster_size.value_prev is None):
            
                # 2.1.1 Create a new point anomaly
                point_anomaly = self._cls_point_anomaly( p_clusters = {cluster.id : cluster},
                                                     p_tstamp = self.get_tstamp(),
                                                     p_visualize = self.get_visualize,
                                                     p_raising_object = self)

                self._raise_anomaly_event( p_anomaly = point_anomaly, p_inst = Instance )
                self._latest_anomaly = point_anomaly
        

## ----------------------------------------------------------------------------------
    def _triage_anomaly( self, p_anomaly : AnomalyCB ):
        """
        Custom method for anomaly triage for point anomalies.
        This method checks if the point anomaly is still valid based on the cluster size.
        """
        # 1 Iterate over all anomalies
       

        if isinstance(p_anomaly, self._cls_point_anomaly):
            #1.1 Get the single cluster associated with the anomaly
            cluster.id, cluster = next(iter(p_anomaly.clusters.items()))

                #1.2 Get the cluster property to be observed
            prop_cluster_size : Property = getattr(cluster, cprop_size_prev)

                # 1.3 Check if the anomaly is still valid, if not remove it
            return prop_cluster_size.value != 1

        return False





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

        
        self.cb_anomalies ={}
        self._property = p_property
        self._cls_spatial_group_anomaly = p_cls_spatial_group_anomaly
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

    def _buffer_anomaly(self, p_anomaly:AnomalyCB):
        """
        Method to be used to add a new anomaly. Please use as part of your algorithm.

        Parameters
        ----------
        p_anomaly : AnomalyCB
            Anomaly object to be added.
        """

        super()._buffer_anomaly(p_anomaly)

        for cluster in p_anomaly.clusters.values():
            self.cb_anomalies[cluster.id] = p_anomaly


## -------------------------------------------------------------------------------------------------
    def _remove_anomaly(self, p_anomaly:AnomalyCB):
        """
        Method to remove an existing anomaly. Please use as part of your algorithm.

        Parameters
        ----------
        p_anomaly : AnomalyCB
            Anomaly object to be removed.
        """

        super()._remove_anomaly(p_anomaly)

        for cluster in p_anomaly.clusters.values():
            del self.cb_anomalies[cluster.id]


## -------------------------------------------------------------------------------------------------
    def _detect_cb_anomalies( self,
                              p_inst: Instance ):

        super()._detect_cb_anomalies( p_inst = p_inst )


        # 1 Get all the clusters from the clusterer
        clusters = self._clusterer.clusters

        # 3 Get average cluster size
        avg_cluster_size = sum([getattr(c, cprop_size[0]).value for c in clusters]) / len(clusters)

        # 4 Calculater the threshold for the cluster size
        self._thres_size = avg_cluster_size * self._thres_percent 


        for cluster in clusters.values():

            # 4.1 Get the cluster property to be observed
            prop_cluster_size : Property = getattr(cluster, self._property)


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
                    self._raise_anomaly_event( p_anomaly = spatial_group_anomaly, p_inst = p_inst )
                    self._latest_anomaly = spatial_group_anomaly

## -------------------------------------------------------------------------------------------------
    def _triage_anomaly( self, p_anomaly : AnomalyCB ):
        """
        Custom method for anomaly triage.
        """
        # 1 Iterate over all anomalies
      

        if isinstance(p_anomaly , self._cls_spatial_group_anomaly):
            #1.1 Get the single cluster associated with the anomaly
            cluster.id, cluster = next(iter(p_anomaly.clusters.items()))

            #1.2 Get the cluster property to be observed
            prop_cluster_size : Property = getattr(cluster, cprop_size_prev)
                
            # 1.3 Check if the anomaly is still valid, if not remove it
            return not (1 < prop_cluster_size.value <= self._thres_size)
                
        return False
            





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
                  p_name : str = None,
                  p_range_max = OAStreamTask.C_RANGE_THREAD,
                  p_ada : bool = True,
                  p_duplicate_data : bool = False,
                  p_visualize : bool = False,
                  p_logging=Log.C_LOG_ALL,
                  p_anomaly_buffer_size : int = 100,
                  **p_kwargs ):
        
        self._cls_temporal_group_anomaly = p_cls_temporal_group_anomaly
        self._temporal_anomalies = {}
        self._tga : TemporalGroupAnomaly = None

        super().__init__( p_clusterer=p_clusterer,
                          p_name = p_name,
                          p_range_max = p_range_max,
                          p_ada = p_ada,
                          p_duplicate_data = p_duplicate_data,
                          p_visualize = p_visualize,
                          p_logging= p_logging,                        
                          **p_kwargs)


## -------------------------------------------------------------------------------------------------
    def _detect_cb_anomaly( self,
                            p_inst: Instance ):

        # Call the parent to detect point and spatial group anomalies
        super()._run_algorithm( p_inst = p_inst )

        
        #1.1 Get the latest anomaly
        latest_anomaly = self._latest_anomaly
    
        #1.2 Process the latest anomaly
        if latest_anomaly is not None:
            #1.2.1 Check for the temporal anomaly conditions
            if isinstance(latest_anomaly, (self._cls_point_anomaly, self._cls_spatial_group_anomaly)) and (latest_anomaly.tstamp == p_inst.tstamp):
                
                # 1.2.2 Check for available temporal anomalies
                if self._tga is None:
                    for cluster_id in latest_anomaly.clusters:
                        if cluster_id not in self._temporal_anomalies:
                            self._temporal_anomalies[cluster_id] = []    
                        self._temporal_anomalies[cluster_id].append(latest_anomaly)

                    if len(self._temporal_anomalies) == 2:
                        temporal_group_anomaly = self._cls_temporal_group_anomaly( p_clusters = {cluster_id : self._clusterer.clusters[cluster_id] for cluster_id in self._temporal_anomalies },
                                                                                   p_status = True,
                                                                                   p_tstamp = p_inst.tstamp,
                                                                                   p_visualize = self.get_visualize,
                                                                                   p_raising_object = self)                                                         
                           
                        self._raise_anomaly_event( p_anomaly = temporal_group_anomaly, p_inst = p_inst )
                        self._tga = temporal_group_anomaly
                        self._temporal_anomalies.clear()
                        self._remove_anomaly(p_anomaly = temporal_group_anomaly)
                           
            
                else:
                    for cluster_id in latest_anomaly.clusters:
                        self._tga.clusters[cluster_id] = self._clusterer.clusters[cluster_id]
                        self._remove_anomaly(p_anomaly = temporal_group_anomaly)
                        self._tga = None

            else:
                if self._tga is None:
                        
                    self._temporal_anomalies.clear()
                    
                else:
                    temporal_group_anomaly = self._cls_temporal_group_anomaly( p_clusters = {cluster_id : self._clusterer.clusters[cluster_id] for cluster_id in self._temporal_anomalies },
                                                                               p_status = False,
                                                                               p_tstamp = p_inst.tstamp,
                                                                               p_visualize = self.get_visualize,
                                                                               p_raising_object = self)                                                         
                            
                    self._raise_anomaly_event( p_anomaly = temporal_group_anomaly, p_inst = p_inst )
                    self._temporal_anomalies.clear()
                    self._tga = None
           

            
## -------------------------------------------------------------------------------------------------
    def _triage_anomaly(self, p_anomaly: AnomalyCB):
        """
        Custom method for anomaly triage.
        """
        # 1 Check if the anomaly is a temporal group anomaly
        if isinstance(p_anomaly, self._cls_temporal_group_anomaly):
            try:
                # 1.1 Get the cluster associated with the anomaly
                cluster_id, cluster = next(iter(p_anomaly.clusters.items()))
            except:
                return False

            # 1.2 Check if the anomaly is still valid
            if cluster.id not in self._temporal_anomalies or len(self._temporal_anomalies[cluster_id]) < 2:
                return False
        
        return True
    