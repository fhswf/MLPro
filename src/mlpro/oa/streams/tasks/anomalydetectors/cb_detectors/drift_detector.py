## -- ----------------------------------------------------------------------------------------------
## -- Project : MLPro - The integrative middleware framework for standardized machine learning
## -- Package : mlpro.oa.tasks.anomalydetectors.cb_detectors
## -- Module  : drift_detector.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2023-06-08  0.0.0     SK       Creation
## -- 2023-09-12  1.0.0     SK       Release
## -- 2024-04-10  1.1.0     DA/SK    Refactoring
## -- 2024-06-20  1.1.1     SK       Bug Fix
## -------------------------------------------------------------------------------------------------

"""
Ver. 1.1.1 (2024-06-20)

This module provides templates for anomaly detection to be used in the context of online adaptivity.
"""

from mlpro.oa.streams.basics import *
from mlpro.oa.streams.tasks.anomalydetectors.cb_detectors.basics import AnomalyDetectorCB
from mlpro.oa.streams.tasks.anomalydetectors.anomalies.clusterbased.drift import ClusterDrift
from mlpro.oa.streams.tasks.clusteranalyzers.basics import ClusterAnalyzer
from mlpro.bf.streams import Instance, InstDict
from mlpro.bf.math.properties import *
from mlpro.oa.streams.tasks.clusteranalyzers.clusters.properties.centroid import cprop_center_geo2
from mlpro.oa.streams.tasks.clusteranalyzers.clusters.properties import cprop_size2
from collections import deque
from sklearn.preprocessing import StandardScaler
import time



## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class ClusterDriftDetector(AnomalyDetectorCB):
    """
    This is the class for detecting change in velocity of clusters.

    """
    C_PROPERTY_DEFINITIONS : PropertyDefinitions = [ cprop_center_geo2,
                                                     cprop_size2]

## -------------------------------------------------------------------------------------------------
    def __init__(self,
                 p_clusterer : ClusterAnalyzer = None,
                 p_velocity_threshold_factor : float = 1.0,
                 p_acceleration_threshld_factor : float = 1.0,
                 p_buffer_size = 5,
                 p_ema_alpha : float = 0.3,
                 p_min_velocity_threshold = 0.01,
                 p_with_time_calculation : bool = True, 
                 p_initial_skip : int = 1,
                 p_name:str = None,
                 p_range_max = StreamTask.C_RANGE_THREAD,
                 p_ada : bool = True,
                 p_duplicate_data : bool = False,
                 p_visualize : bool = False,
                 p_logging=Log.C_LOG_ALL,
                 **p_kwargs):

        super().__init__(p_clusterer = p_clusterer,
                         p_name = p_name,
                         p_range_max = p_range_max,
                         p_ada = p_ada,
                         p_duplicate_data = p_duplicate_data,
                         p_visualize = p_visualize,
                         p_logging = p_logging,
                         **p_kwargs)
        
        for x in self.C_PROPERTY_DEFINITIONS:
            if x not in self.C_REQ_CLUSTER_PROPERTIES:
                self.C_REQ_CLUSTER_PROPERTIES.append(x)

        unknown_prop = self._clusterer.align_cluster_properties(p_properties=self.C_REQ_CLUSTER_PROPERTIES)

        #if len(unknown_prop) > 0:
        #    raise RuntimeError("The following cluster properties need to be provided by the clusterer: ", unknown_prop)


        # Parameters for dynamic threshold, EMA smoothing, and time-based calculation
        self._velocity_thresh_factor = p_velocity_threshold_factor
        self._acceleration_thresh_factor = p_acceleration_threshld_factor
        self._buffer_size = p_buffer_size
        self._ema_alpha = p_ema_alpha
        self._min_velocity_threshold = p_min_velocity_threshold
        self._with_time_calculation = p_with_time_calculation
        self._init_skip = p_initial_skip
        self._visualize = p_visualize
        self._count = 0
        self._count_change = {}
        
        # Data structures for storing previous states and buffers
        self._centroids_history = {}
        self._times_history = {}
        self._velocities_history = {}
        self._accelerations_history = {}
        self._velocities = {}
        self._accelerations = {}
        self._scaler = StandardScaler()
        self._cluster_states = {}
        self._velocity_threshold = {}
        self._acceleration_threshold = {}


## -------------------------------------------------------------------------------------------------
    def _run(self, p_inst : InstDict):
        new_instances = []
        for inst_id, (inst_type, inst) in sorted(p_inst.items()):
            new_instances.append(inst)

        clusters = self._clusterer.get_clusters()

        drifting_clusters = {}

        for id, cluster in clusters.items():
            current_centroid = np.array(cluster.centroid.value)
            current_time = time.time()  # Get the current timestamp

            # Initialize cluster if it does not exist
            if id not in self._centroids_history:
                self._initialize_cluster(id, current_centroid, current_time)

            # Transform the current centroid using the scaler
            #self._scaler.partial_fit(current_centroid)
            #current_centroid = self._scaler.transform(current_centroid)
            #current_centroid = self._scaler.fit_transform(current_centroid)[0]
            previous_centroid = self._centroids_history[id]
            if self._with_time_calculation:
                previous_time = self._times_history[id]
            else:
                previous_time = None
            
            # Calculate velocity
            velocity = self._calculate_velocity(previous_centroid, current_centroid, previous_time, current_time, cluster_id=id)

            if velocity:
                # Calculate average velocity
                average_velocity = self._calculate_avg(self._velocities_history[id])
                self._velocities[id] = average_velocity

                # Calculate acceleration
                acceleration = self._calculate_acceleration(self._velocities_history[id], previous_time, current_time, cluster_id=id)
                if acceleration:
                    #acceleration = np.nan_to_num(acceleration)
                    average_acceleration = self._calculate_avg(self._accelerations_history[id])
                    self._accelerations[id] = average_acceleration

                # Update previous centroid and time
                self._centroids_history[id] = current_centroid
                if self._with_time_calculation:
                    self._times_history[id] = current_time
        
                # Clean the velocity to ensure no NaNs or infinities
                #velocity = np.nan_to_num(velocity)

                # Detect state changes and anomalies
                state_change = self._detect_state_change(id, average_velocity, average_acceleration, self._velocity_threshold[id], self._acceleration_threshold[id])
            
                if state_change:
                    drifting_clusters[id] = cluster

                # Calculate dynamic thresholds
                self._velocity_threshold[id] = self._dynamic_threshold(self._velocity_threshold[id], self._velocities_history[id][-1], self._velocity_thresh_factor)
                self._acceleration_threshold[id] = self._dynamic_threshold(self._acceleration_threshold[id], self._accelerations_history[id][-1], self._acceleration_thresh_factor)


        if (self._count >= self._init_skip):
            if len(drifting_clusters) != 0:
                anomaly = ClusterDrift(p_id = self._get_next_anomaly_id,
                                 p_instances=new_instances,
                                 p_clusters=drifting_clusters,
                                 p_det_time=str(inst.get_tstamp()),
                                 p_visualize=self._visualize)
                self._raise_anomaly_event(anomaly)

        if self._count < self._init_skip:
            self._count += 1
            

    ## -------------------------------------------------------------------------------------------------
    def _initialize_cluster(self, cluster_id, current_centroid, timestamp):
        # Initialize cluster state and data structures
        self._centroids_history[cluster_id] = current_centroid         
        self._velocities_history[cluster_id] = np.zeros((self._buffer_size, current_centroid.shape[0]))
        self._accelerations_history[cluster_id] = np.zeros((self._buffer_size, current_centroid.shape[0]))
        self._velocities[cluster_id] = np.zeros(current_centroid.shape[0])
        self._accelerations[cluster_id] = np.zeros(current_centroid.shape[0])
        self._cluster_states[cluster_id] = 'initial'
        self._count_change[cluster_id] = 0
        if self._with_time_calculation:
            self._times_history[cluster_id] = timestamp

        self._velocity_threshold[cluster_id] = np.full(current_centroid.shape[0], self._velocity_thresh_factor)
        self._acceleration_threshold[cluster_id] = np.full(current_centroid.shape[0], self._acceleration_thresh_factor)


    ## -------------------------------------------------------------------------------------------------
    def _calculate_velocity(self, previous_centroid, current_centroid, previous_time, current_time, cluster_id):
        # Calculate the difference in centroids to get velocity
        difference = current_centroid - previous_centroid
        
        # Calculate time-based velocity if enabled
        if self._with_time_calculation and previous_time is not None and current_time is not None:
            time_diff = current_time - previous_time
            if not np.all(difference==0):
                if time_diff > 0:
                    velocity = difference / time_diff
                    self._velocities_history[cluster_id] = np.append(self._velocities_history[cluster_id][1:], [velocity], axis=0)
                    return True
                else:
                    return False
            else:
                return False
            
        else:
            if np.all(difference==0):
                self._count_change[cluster_id] += 1
                return False
            else:
                velocity = difference/self._count_change[cluster_id]
                self._count_change[cluster_id] = 1
                self._velocities_history[cluster_id] = np.append(self._velocities_history[cluster_id][1:], [velocity], axis=0)
                return True

    ## -------------------------------------------------------------------------------------------------
    def _calculate_acceleration(self, velocity_buffer, previous_time, current_time, cluster_id):
        # Calculate acceleration as the change in velocity
        if len(velocity_buffer) < 2:
            return False
        
        acceleration = velocity_buffer[-1] - velocity_buffer[-2]
        
        # Calculate time-based acceleration if enabled
        if self._with_time_calculation and previous_time is not None and current_time is not None:
            time_diff = current_time - previous_time
            if time_diff > 0:
                acceleration /= time_diff
                self._accelerations_history[cluster_id] = np.append(self._accelerations_history[cluster_id][1:], [acceleration], axis=0)
                return True
            else:
                return False
        else:
            self._accelerations_history[cluster_id] = np.append(self._accelerations_history[cluster_id][1:], [acceleration], axis=0)
            return True


    ## -------------------------------------------------------------------------------------------------
    def _calculate_avg(self, buffer):
        if len(buffer) == 0:
            return np.zeros(buffer[0].shape)
        
        return np.mean(buffer, axis=0)


    ## -------------------------------------------------------------------------------------------------
    def _calculate_ema(self, buffer):
        # Calculate Exponential Moving Average (EMA) for smoothing
        if len(buffer) == 0:
            return np.zeros(buffer[0].shape)

        ema = np.zeros(buffer[0].shape)
        print(buffer)
        print(ema)
        alpha = self._ema_alpha
        for i, value in enumerate(buffer):
            #value = np.nan_to_num(value, nan=0.0, posinf=1e6, neginf=-1e6)  # Ensure no NaNs or infinities
            ema = alpha * np.float64(value) + (1 - alpha) * ema if i > 0 else value
            print(ema)
        return ema
    

    ## -------------------------------------------------------------------------------------------------
    def _dynamic_threshold(self, threshold, value, threshold_factor):
        # Calculate dynamic threshold
        return threshold*(1-self._ema_alpha) + self._ema_alpha*value*threshold_factor


    ## -------------------------------------------------------------------------------------------------
    def _detect_state_change(self, cluster_id, average_velocity, average_acceleration, velocity_threshold, acceleration_threshold):
        # Detect various states of cluster behavior
        previous_state = self._cluster_states[cluster_id]
        
        if previous_state == 'initial' and np.linalg.norm(average_velocity) > self._min_velocity_threshold:
            self._cluster_states[cluster_id] = 'moving'
            return True
        
        if previous_state == 'moving' and np.linalg.norm(average_velocity) < self._min_velocity_threshold:
            self._cluster_states[cluster_id] = 'stopped'
            return True
        
        if previous_state == 'stopped' and np.linalg.norm(average_velocity) > self._min_velocity_threshold:
            self._cluster_states[cluster_id] = 'moving'
            return True
        
        if np.linalg.norm(average_acceleration) > np.linalg.norm(acceleration_threshold):
            self._cluster_states[cluster_id] = 'accelerating'
            return True
        
        if previous_state == 'moving' and np.linalg.norm(average_velocity - self._velocities[cluster_id]) > np.linalg.norm(velocity_threshold):
            self._cluster_states[cluster_id] = 'velocity_change'
            return True
        
        return False


    ## -------------------------------------------------------------------------------------------------
    def get_velocities(self):
        return self._velocities


    ## -------------------------------------------------------------------------------------------------
    def get_accelerations(self):
        return self._accelerations
    