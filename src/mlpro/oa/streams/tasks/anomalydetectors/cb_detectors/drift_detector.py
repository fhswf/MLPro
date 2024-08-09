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
## -- 2024-05-28  1.2.0     SK       Refactoring
## -------------------------------------------------------------------------------------------------

"""
Ver. 1.2.0 (2024-05-28)

This module provides cluster drift detector algorithm.
"""

from mlpro.oa.streams.basics import *
from mlpro.oa.streams.tasks.anomalydetectors.cb_detectors.basics import AnomalyDetectorCB
from mlpro.oa.streams.tasks.anomalydetectors.anomalies.clusterbased.drift import ClusterDrift
from mlpro.oa.streams.tasks.clusteranalyzers.basics import ClusterAnalyzer
from mlpro.bf.streams import Instance, InstDict
from mlpro.bf.math.properties import *
from mlpro.oa.streams.tasks.clusteranalyzers.clusters.properties.centroid import cprop_centroid2
import time



## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class ClusterDriftDetector(AnomalyDetectorCB):
    """
    This is the class for detecting change in velocity of clusters.

    """
    C_PROPERTY_DEFINITIONS : PropertyDefinitions = [cprop_centroid2]

## -------------------------------------------------------------------------------------------------
    def __init__(self,
                 p_clusterer : ClusterAnalyzer = None,
                 p_instantaneous_velocity_change_detection : bool = True,
                 p_min_velocity_threshold : float = 0.01,
                 p_state_change_detection : bool = False,
                 p_min_acceleration_threshold : float = False,
                 p_buffer_size = 5,
                 p_ema_alpha : float = 0.7,
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
        
        # Parameters for dynamic threshold, EMA smoothing, and time-based calculation
        self._inst_change_det = p_instantaneous_velocity_change_detection
        self._state_change_det = p_state_change_detection
        self._min_vel_thresh = p_min_velocity_threshold
        self._min_acc_thresh = p_min_acceleration_threshold
        self._buffer_size = p_buffer_size
        self._ema_alpha = p_ema_alpha
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
        self._cluster_states = {}


## -------------------------------------------------------------------------------------------------
    def _run(self, p_inst : InstDict):
        new_instances = []
        for inst_id, (inst_type, inst) in sorted(p_inst.items()):
            new_instances.append(inst)

        clusters = self._clusterer.get_clusters()

        drifting_clusters = {}
        properties = {}

        for id, cluster in clusters.items():
            current_centroid = np.array(cluster.centroid.value)
            current_time = time.time()  # Get the current timestamp

            # Initialize cluster if it does not exist
            if id not in self._centroids_history:
                self._initialize_cluster(id, current_centroid, current_time)

            previous_centroid = self._centroids_history[id]
            if self._with_time_calculation:
                previous_time = self._times_history[id]
            else:
                previous_time = None
            
            # Calculate velocity
            velocity = self._calculate_velocity(previous_centroid, current_centroid, previous_time, current_time, cluster_id=id)
            
            if velocity:
                drift = False
                state_change = False
                # Calculate average velocity
                self._calculate_ema(id)

                # Calculate acceleration
                self._calculate_acceleration(self._velocities_history[id], previous_time, current_time, cluster_id=id)

                # Update previous centroid and time
                self._centroids_history[id] = current_centroid
                if self._with_time_calculation:
                    self._times_history[id] = current_time
                else:
                    self._count_change[id] = 1

                # Detection of instantaneous velocity change
                if self._inst_change_det:
                    drift = self._inst_vel_change_detection(self._velocities_history[id][-1], self._min_vel_thresh)

                else:
                    # Detect state changes and anomalies
                    state_change = self._detect_state_change(id, self._velocities_history[id], self._accelerations_history[id], self._min_vel_thresh, self._min_acc_thresh)
                
                if drift or state_change:
                        drifting_clusters[id] = cluster
                        properties[id] = {"velocity":np.linalg.norm(self._velocities_history[id][-1]), "acceleration":np.linalg.norm(self._accelerations_history[id][-1])}

        # Raise Anomaly event
        if (self._count >= self._init_skip):
            if len(drifting_clusters) != 0:
                anomaly = ClusterDrift(p_id = self._get_next_anomaly_id,
                                 p_instances=new_instances,
                                 p_clusters=drifting_clusters,
                                 p_properties=properties,
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
        self._cluster_states[cluster_id] = 'initial'
        self._count_change[cluster_id] = 0
        if self._with_time_calculation:
            self._times_history[cluster_id] = timestamp


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
                
                self._velocities_history[cluster_id] = np.append(self._velocities_history[cluster_id][1:], [velocity], axis=0)
                return True

    ## -------------------------------------------------------------------------------------------------
    def _calculate_acceleration(self, velocity_buffer, previous_time, current_time, cluster_id):
        # Calculate acceleration as the change in velocity
        acceleration = velocity_buffer[-1] - velocity_buffer[-2]
        
        # Calculate time-based acceleration if enabled
        if self._with_time_calculation:
            time_diff = current_time - previous_time
            acceleration /= time_diff
            self._accelerations_history[cluster_id] = np.append(self._accelerations_history[cluster_id][1:], [acceleration], axis=0)
        else:
            acceleration = acceleration/self._count_change[cluster_id]
            self._accelerations_history[cluster_id] = np.append(self._accelerations_history[cluster_id][1:], [acceleration], axis=0)


    ## -------------------------------------------------------------------------------------------------
    def _inst_vel_change_detection(self, velocity, threshold):
        if np.linalg.norm(velocity) >= threshold:
            return True
        else:
            return False

        
    ## -------------------------------------------------------------------------------------------------
    def _calculate_ema(self, cluster_id):
        # Calculate Exponential Moving Average (EMA) for smoothing
        ema = self._velocities_history[cluster_id][-2]
        alpha = self._ema_alpha
        if np.all(self._velocities_history[cluster_id][-2] == 0):
            velocity = self._velocities_history[cluster_id][-1]
        else:
            velocity = alpha * self._velocities_history[cluster_id][-1] + (1 - alpha) * self._velocities_history[cluster_id][-2]
    
        self._velocities_history[cluster_id] = np.append(self._velocities_history[cluster_id][:-1], [velocity], axis=0)
    

    ## -------------------------------------------------------------------------------------------------
    def _detect_state_change(self, cluster_id, average_velocity_buffer, average_acceleration_buffer, vel_thresh, acc_thresh):
        # Detect various states of cluster behavior
        previous_state = self._cluster_states[cluster_id]

        velocity_norms = np.linalg.norm(average_velocity_buffer, axis=1)
        acceleration_norms = norms = np.linalg.norm(average_acceleration_buffer, axis=1)

        detected = False

        if vel_thresh and acc_thresh:
            if (previous_state in['initial', 'stopped']) and np.any(abs(velocity_norms) > vel_thresh):
                self._cluster_states[cluster_id] = 'moving'
                detected = True
            
            if (previous_state in ['moving', 'accelerating']) and np.all(abs(velocity_norms) < vel_thresh):
                self._cluster_states[cluster_id] = 'stopped'
                detected = True

            if (previous_state in ['initial', 'stopped', 'moving']) and any(abs(acceleration_norms) > acc_thresh):
                self._cluster_states[cluster_id] = 'accelerating'
                detected = True
        else:
            if vel_thresh:
                if (previous_state in['initial', 'stopped']) and np.any(abs(velocity_norms) > vel_thresh):
                    self._cluster_states[cluster_id] = 'moving'
                    detected = True
                
                if previous_state == 'moving' and np.all(abs(velocity_norms) < vel_thresh):
                    self._cluster_states[cluster_id] = 'stopped'
                    detected = True

            if acc_thresh:
                if (previous_state in ['initial', 'stopped']) and any(abs(acceleration_norms) > acc_thresh):
                    self._cluster_states[cluster_id] = 'accelerating'
                    detected = True
                
                if previous_state == 'accelerating' and all(abs(acceleration_norms) < acc_thresh):
                    self._cluster_states[cluster_id] = 'stopped'
                    detected = True

        return detected


    ## -------------------------------------------------------------------------------------------------
    def get_velocities(self):
        return self._velocities_history


    ## -------------------------------------------------------------------------------------------------
    def get_accelerations(self):
        return self._accelerations_history
    