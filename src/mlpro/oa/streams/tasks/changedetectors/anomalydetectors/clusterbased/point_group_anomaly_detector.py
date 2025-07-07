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
## -- 2025-06-02  0.3.1     DS       Bug fixes
## -- 2025-06-08  0.3.2     DS       Design extensions
## -- 2025-06-10  0.3.3     DA/DS    Refactoring
## -- 2025-06-15  0.3.4     DS       Bug fixes
## -- 2025-06-27  0.3.5     DS       Bug fixes
## -- 2025-06-30  0.3.6     DS       Bug fixes
## -------------------------------------------------------------------------------------------------

"""
Ver. 0.3.6 (2025-06-27)

This module provides cluster based point and group anomaly detector algorithm.
"""

from mlpro.bf.various import Log
from mlpro.bf.math.properties import Property, PropertyDefinition
from mlpro.bf.streams import Instance
from mlpro.oa.streams import OAStreamTask
from mlpro.oa.streams.tasks.clusteranalyzers import ClusterAnalyzer
from mlpro.oa.streams.tasks.clusteranalyzers.clusters.properties import cprop_size, cprop_size_prev
from mlpro.oa.streams.tasks.changedetectors.anomalydetectors.clusterbased.basics import AnomalyDetectorCB
from mlpro.oa.streams.tasks.changedetectors.anomalydetectors.anomalies.clusterbased import AnomalyCB, PointAnomaly, SpatialGroupAnomaly, TemporalGroupAnomaly


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
    p_thrs_inst : int = 0
        The threshold for the number of instances.
    p_thrs_clusters : int = 1
        The threshold for the number of clusters.
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
                  p_thrs_inst : int = 0,
                  p_thrs_clusters : int = 1,
                  **p_kwargs ):
        
        if isinstance(p_property, tuple):
            p_property = p_property[0]
        self._property          = p_property
        self._cls_point_anomaly = p_cls_point_anomaly
        self._latest_anomaly    = None
        
        super().__init__( p_clusterer=p_clusterer,
                          p_name = p_name,
                          p_range_max = p_range_max,
                          p_ada = p_ada,
                          p_duplicate_data = p_duplicate_data,
                          p_visualize = p_visualize,
                          p_logging= p_logging,
                          p_anomaly_buffer_size = p_anomaly_buffer_size,
                          p_thrs_inst = p_thrs_inst,
                          p_thrs_clusters = p_thrs_clusters,                       
                          **p_kwargs)


## -------------------------------------------------------------------------------------------------  
    def _detect(self, p_clusters, p_instance, **p_kwargs):
        """
        custom method for detectiong cluster-based point anomalies.
        """
        # 1 Call the parent method to detect anomalies
        for cluster_id, cluster in p_clusters.items():

        # 2 Get the cluster property to be observed
            prop_cluster_size : Property = getattr(cluster, self._property)
 
            #curr_cluster_size = prop_cluster_size.value if prop_cluster_size.value is not None else 0
            #prev_cluster_size = prop_cluster_size.value_prev if prop_cluster_size.value_prev is not None else 0

            #if curr_cluster_size == 1 and prev_cluster_size is None or prev_cluster_size > 1:


            # 2.1 Check for the point anomalies
            if (prop_cluster_size.value == 1) and (prop_cluster_size.value_prev is None or prop_cluster_size.value_prev == 0 ):

                # 2.1.1 Check if the cluster is already in the anomaly buffer
                if cluster_id in self.cb_anomalies:
                    continue
            
                    # 2.1.1 Create a new point anomaly
                point_anomaly = self._cls_point_anomaly( p_clusters = {cluster_id: cluster},
                                                         p_tstamp = self._get_tstamp(),
                                                         p_visualize = self.get_visualization(),
                                                         p_raising_object = self)

                self._raise_anomaly_event( p_anomaly = point_anomaly, p_instance = p_instance )
                self._latest_anomaly = point_anomaly
                
            #else:
                #self.log(Log.C_LOG_TYPE_I, f"No anomaly: Cluster {cluster_id} size {prop_cluster_size.value} (prev: {prop_cluster_size.value_prev})")
        

## ----------------------------------------------------------------------------------
    def _triage_anomaly( self, p_anomaly : AnomalyCB, **p_kwargs ):
        """
        Custom method for anomaly triage for point anomalies.
        This method checks if the point anomaly is still valid based on the cluster size.
        """
        # 1 Iterate over all anomalies
        if isinstance(p_anomaly, self._cls_point_anomaly):
            #1.1 Get the single cluster associated with the anomaly
            cluster = next(iter(p_anomaly.clusters.values()))


                #1.2 Get the cluster property to be observed
            prop_cluster_size : Property = getattr(cluster, cprop_size_prev[0])

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
    P_thres_inst : int = 0
        The threshold for the number of instances.
    p_thres_clusters : int = 1
        The threshold for the number of clusters.
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
                  P_thres_inst : int = 0,
                  p_thres_clusters : int = 1,
                  p_thres_percent : float = 0.05,
                  **p_kwargs ):

        self._property                  = p_property
        self._cls_spatial_group_anomaly = p_cls_spatial_group_anomaly
        self._thres_percent             = p_thres_percent
        self.clusters_to_remove         = []
        
        super().__init__( p_clusterer=p_clusterer,
                          p_name = p_name,
                          p_range_max = p_range_max,
                          p_ada = p_ada,
                          p_duplicate_data = p_duplicate_data,
                          p_visualize = p_visualize,
                          p_logging= p_logging,   
                          p_anomaly_buffer_size = p_anomaly_buffer_size,
                          p_thrs_inst = P_thres_inst,
                          p_thrs_clusters = p_thres_clusters,                     
                          **p_kwargs)


## -------------------------------------------------------------------------------------------------
    def _detect(self, p_clusters : dict, p_instance : Instance, **p_kwargs):
        """
        Custom method for detecting cluster-based spatial group anomalies.
        """

        super()._detect( p_clusters = p_clusters, p_instance = p_instance, **p_kwargs)

        # 1 Get average cluster size
        cluster_sizes = [getattr(c, cprop_size[0]).value 
                         for c in p_clusters.values()
                           if getattr(c,cprop_size[0]) is not None and getattr(c, cprop_size[0]).value is not None
                        ]
        avg_cluster_size = sum(cluster_sizes)/ len(p_clusters) if len(p_clusters) > 0 else 0
        # 2 Calculater the threshold for the cluster size
        self._thres_size = avg_cluster_size * self._thres_percent 


        for cluster in p_clusters.values():

            # 2.1 Get the cluster property to be observed
            prop_cluster_size : Property = getattr(cluster, self._property)

            if prop_cluster_size.value is None:
                continue
            # 2.2 Check for the  spatial group anomalies
            if 1 < prop_cluster_size.value <= self._thres_size:

                # 2.2.1 Create a new spatial group anomaly 
                try:
                    cb_anomaly = self.cb_anomalies[cluster.id]

                    if  type(cb_anomaly) == self._cls_point_anomaly:
                        self._remove_anomaly(p_anomaly = cb_anomaly)
                        create_anomaly = True
                    else:
                        create_anomaly = False
                        
                except:

                    create_anomaly = True

                # 2.2.2 Create a new spatial group anomaly    
                if create_anomaly:
                    spatial_group_anomaly = self._cls_spatial_group_anomaly( p_clusters = {cluster.id : cluster},
                                                                             p_tstamp = self._get_tstamp(),
                                                                             p_visualize = self.get_visualization,
                                                                             p_raising_object = self)

                    # 2.2.2.1 Raise an anomaly event
                    self._raise_anomaly_event( p_anomaly = spatial_group_anomaly, p_instance = p_instance )
                    self._latest_anomaly = spatial_group_anomaly


## -------------------------------------------------------------------------------------------------
    def _triage_anomaly( self, p_anomaly : AnomalyCB, **p_kwargs ):
        """
        Custom method for anomaly triage for spatial group anomalies.
        This method checks if the spatial group anomaly is still valid based on the cluster size.
        """

        # 1 Iterate over all anomalies
        if isinstance(p_anomaly , self._cls_spatial_group_anomaly):
            #1.1 Get the single cluster associated with the anomaly
            cluster = next(iter(p_anomaly.clusters.values()))

            #1.2 Get the cluster property to be observed
            prop_cluster_size : Property = getattr(cluster, cprop_size_prev[0])

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
    P_thrs_inst : int = 0
        The threshold for the number of instances.
    p_thrs_clusters : int = 1
        The threshold for the number of clusters.
    """

    C_NAME = 'Temporal group anomaly'
    C_PROPERTY_DEFINITIONS : PropertyDefinition = []

## -------------------------------------------------------------------------------------------------
    def __init__( self,
                  p_clusterer : ClusterAnalyzer,
                  p_cls_temporal_group_anomaly : type = TemporalGroupAnomaly,
                  p_name : str = None,
                  p_range_max = OAStreamTask.C_RANGE_THREAD,
                  p_ada : bool = True,
                  p_duplicate_data : bool = False,
                  p_visualize : bool = False,
                  p_logging=Log.C_LOG_ALL,
                  p_anomaly_buffer_size : int = 100,
                  P_thrs_inst : int = 0,
                  p_thrs_clusters : int = 1,
                  **p_kwargs ):
        
        self._cls_temporal_group_anomaly = p_cls_temporal_group_anomaly
        self._tga : TemporalGroupAnomaly = None

        super().__init__( p_clusterer=p_clusterer,
                          p_name = p_name,
                          p_range_max = p_range_max,
                          p_ada = p_ada,
                          p_duplicate_data = p_duplicate_data,
                          p_visualize = p_visualize,
                          p_logging= p_logging, 
                          p_anomaly_buffer_size = p_anomaly_buffer_size,
                          p_thrs_inst = P_thrs_inst,
                          p_thrs_clusters = p_thrs_clusters,                       
                          **p_kwargs)


## -------------------------------------------------------------------------------------------------
    def _detect(self, p_clusters : dict, p_instance : Instance, **p_kwargs):

        #1 Call the parent to detect point and spatial group anomalies
        super()._detect( p_clusters = p_clusters, p_instance = p_instance, **p_kwargs)

        #2 Get the latest anomaly
        latest_anomaly = self._latest_anomaly
    
        #3 Process the latest anomaly
        if latest_anomaly is None:

            return
        
        #3.1 Check for the temporal anomaly conditions
        if isinstance(latest_anomaly, (self._cls_point_anomaly, self._cls_spatial_group_anomaly)) and (latest_anomaly.tstamp == p_instance.tstamp):
                
            #3.1.1 If temporal anomaly does not exist, create a new one and add clusters of the latest anomaly to _tga
            if self._tga is None:

                self._tga = self._cls_temporal_group_anomaly(p_clusters = latest_anomaly.clusters,
                                                             p_status = True,
                                                             p_tstamp = p_instance.tstamp,
                                                             p_visualize = self.get_visualize,
                                                             p_raising_object = self)
                
            self._tga.add_clusters(p_clusters= latest_anomaly.clusters)
            
            #3.1.2 If the temporal anomaly has only two clusters, raise an anomaly event
            if len(self._tga.clusters) == 2:
                
                self._raise_anomaly_event(p_anomaly = self._tga, 
                                          p_instance = p_instance)

            return    
        
        #3.2 If temporal anomaly conditiod is not met, 
        elif self._tga is None:
            return
               
        if len(self._tga.clusters) >= 2:
                    
            self._tga.status = False
            self._raise_anomaly_event(p_anomaly=self._tga, p_instance = p_instance)

        self._tga = None

                    
                  
## -------------------------------------------------------------------------------------------------
    def _triage_anomaly( self, p_anomaly : AnomalyCB, **p_kwargs ):
        """
        Custom method for anomaly triage for temporal group anomalies.
        This method checks if the temporal group anomaly is still valid.
        """
        # 1 Check if the anomaly is a temporal group anomaly
        if isinstance(p_anomaly, self._cls_temporal_group_anomaly):
            try:
                # 1.1 Get the cluster associated with the anomaly
                cluster = next(iter(p_anomaly.clusters.values()))
            except:
                return False

            # 1.2 Check if the anomaly is still valid
            if cluster.id not in self._temporal_anomalies or len(self._temporal_anomalies[cluster.id]) < 2:
                return False
        
        return True
    