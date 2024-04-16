## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - A Synoptic Framework for Standardized Machine Learning Tasks
## -- Package : mlpro.bf.streams.streams
## -- Module  : cluster_generator.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2023-11-20  0.0.0     SK       Creation
## -- 2024-01-02  1.0.0     SK       First draft implementation
## -- 2024-03-24  1.0.0     SK       Completion of class documentations
## -------------------------------------------------------------------------------------------------

"""
Ver. 1.0.0 (2024-03-24)

This module provides the native stream class StreamMLProCloudsAdvanced.
These stream provides instances with self.C_NUM_DIMENSIONS dimensional random feature data, placed around
centers (can be defined by user) which may or maynot move over time.

"""


import random
import math
from mlpro.bf.streams.models import *
from mlpro.bf.streams.streams.provider_mlpro import StreamMLProBase



## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class StreamMLProCloudsAdvanced (StreamMLProBase):
    """
    This benchmark stream class generates freely configurable random point clouds of any number, size
    and dimensionality. Optionally, the centers of the clouds are static or in motion.

    Parameters
    ----------
    p_num_dim : int
        The number of dimensions or features of the data. Default = 3.
    p_num_instances : int
        Total number of instances. The value '0' means indefinite. Default = 1000.
    p_num_clouds : int
        Number of clouds. Default = 4.
    p_radii : list
        Radii of the clouds. Default = 100.
    p_weights : list[]
        Optional list of integer weights per cloud. For example, a list [1,2] causes the second cloud 
        to be flooded with two times more instances than the first one. If empty or None, all clouds 
        are flooded randomly but equally.
    p_velocity : foat
        Velocity for the centers in unit 1/di. Default = 0.0.
    p_seed 
        Seeding value for the random generator. Default = None (no seeding).
    p_logging
        Log level (see constants of class Log). Default: Log.C_LOG_ALL.
    """

    C_ID                    = 'AnomalyCloudsNDim'
    C_NAME                  = 'Clouds N-Dim'
    C_TYPE                  = 'Benchmark'
    C_VERSION               = '1.0.0'
    C_SCIREF_ABSTRACT       = 'Demo stream provides self.C_NUM_INSTANCES C_NUM_DIMENSIONS-dimensional instances per cluster randomly positioned around centers which may or maynot move over time.'
    C_BOUNDARIES            = [-1000,1000]

## -------------------------------------------------------------------------------------------------
    def __init__( self,
                  p_num_dim : int = 2,
                  p_num_instances : int = 1000,
                  p_num_clouds : int = 4,
                  p_radii : list = [100.0],
                  p_weights : list = [],
                  p_velocity : float = 0.0,
                  p_seed = None,
                  p_logging = Log.C_LOG_ALL,
                  **p_kwargs ):
        
        self._num_dim         = p_num_dim
        self._radii           = p_radii
        self._num_clouds      = int(p_num_clouds)
        self._velocity        = p_velocity
        self._weights         = p_weights
        self._cloud_ids       = []
        self._num_cloud_ids   = 0
        self._centers         = []
        self._centers_step    = []

        self.C_NUM_INSTANCES  = p_num_instances

        self.set_random_seed(p_seed=p_seed)

        if ( self._weights is not None ) and ( len(self._weights) != 0 ):
            if len(self._weights) != p_num_clouds:
                raise ParamError('The number of weights (parameter p_weights) needs to be equal to the number of clouds (parameter p_num_clouds)')
            
            for c, weight in enumerate(self._weights):
                for w in range(weight):
                    self._cloud_ids.append(c)

            self._num_cloud_ids = len(self._cloud_ids)

        StreamMLProBase.__init__ (self,
                                  p_logging=p_logging,
                                  **p_kwargs)
        

## -------------------------------------------------------------------------------------------------
    def _setup_feature_space(self) -> MSpace:
        feature_space : MSpace = MSpace()

        for i in range(self._num_dim):
            feature_space.add_dim( Feature( p_name_short = 'f_' + str(i),
                                            p_base_set = Feature.C_BASE_SET_R,
                                            p_name_long = 'Feature #' + str(i),
                                            p_name_latex = '',
                                            p_description = '',
                                            p_symmetrical = False,
                                            p_logging=Log.C_LOG_NOTHING ) )

        return feature_space


## -------------------------------------------------------------------------------------------------
    def _init_dataset(self):

        # 1 Compute the initial positions of the centers
        for c in range(self._num_clouds):

            center = np.zeros(self._num_dim)

            for d in range(self._num_dim):
                center[d] = random.randint(self.C_BOUNDARIES[0], self.C_BOUNDARIES[1])

            self._centers.append(center)


        # 2 Commpute a vectorial step for each center based on the given velocity
        if self._velocity != 0.0:

            for c in range(self._num_clouds):

                center_step = np.zeros(self._num_dim)
                dist = 0

                for d in range(self._num_dim):
                    center_step[d] = random.randint(self.C_BOUNDARIES[0], self.C_BOUNDARIES[1])
                    dist = dist + center_step[d]**2

                dist = dist**0.5
                f    = self._velocity / dist

                for d in range(self._num_dim):
                    center_step[d] *= f 
                    
                self._centers_step.append(center_step)


## -------------------------------------------------------------------------------------------------
    def _get_next(self) -> Instance:

        # 0 Preparation
        if self.C_NUM_INSTANCES== 0: pass
        elif self._index == self.C_NUM_INSTANCES: raise StopIteration


        # 1 Update of center positions
        if self._velocity != 0.0:
            for c in range(self._num_clouds):
                self._centers[c] = self._centers[c] + self._centers_step[c]


        # 2 Determination of the cloud to be fed
        if self._num_cloud_ids == 0:
            # 2.1 Next cloud is found randomly
            c = random.randint(0, self._num_clouds - 1)

        else:
            # 2.1 Next cloud is found by user requirement (see parameter p_weights)
            c_id = random.randint(0, self._num_cloud_ids - 1)
            c    = self._cloud_ids[c_id]


        # 3 Generation of random point around a randomly selected center
        center       = self._centers[c]
        feature_data = Element(self._feature_space)
        point_values = np.zeros(self._num_dim)

        if len(self._radii) == 1:
            radius = self._radii[0]
        else:
            radius = self._radii[c]

        if self._num_dim == 2:
            # 3.1 Generation of a random 2D point within a circle around the center
            radian          = random.random() * 2 * math.pi
            radius_rnd      = radius * random.random()
            point_values[0] = center[0] + math.cos(radian) * radius_rnd
            point_values[1] = center[1] + math.sin(radian) * radius_rnd
        elif self._num_dim == 3:
            # 3.2 Generation of a random 3D point within a sphere around the center
            radian1         = random.random() * 2 * math.pi
            radian2         = random.random() * 2 * math.pi
            radius_rnd      = radius * random.random()
            point_values[0] = center[0] + math.cos(radian1) * math.cos(radian2) * radius_rnd
            point_values[1] = center[1] + math.sin(radian2) * radius_rnd
            point_values[2] = center[2] + math.sin(radian1) * math.cos(radian2) * radius_rnd
        else:
            # 3.3 Generation of a random nD point in a hypercube with edge length (2 * radius) around the center
            for d in range(self._num_dim):
                point_values[d] = center[d] + random.randint(0, 2 * radius) - radius

        feature_data.set_values(point_values)

        self._index += 1

        return Instance( p_feature_data=feature_data )


import numpy as np

class ClusterGenerator:
    def __init__(self, num_dimensions, num_clusters, num_points_per_cluster=1000, num_point_anomalies=10):
        self.num_dim = num_dimensions
        self.num_clusters = num_clusters
        self.num_points_per_cluster = num_points_per_cluster
        self.clusters = []
        self.num_point_anom = num_point_anomalies
        self.index = 0
        self.cluster = 0
        self.cycle = 0

    def add_cluster(self, center=None,
                    radius = 100,
                    velocity=0,
                    change_in_radius=False,
                    change_in_velocity=False,
                    change_in_density=False,
                    appears_later=False,
                    disappears=False,
                    merge=False):
        cluster = {
            "dimension": self.num_dim,
            "center": center,
            "num_points": self.num_points_per_cluster,
            "radius" : radius,
            "velocity": velocity,
            "change_in_radius" : change_in_radius,
            "change_in_velocity" : change_in_velocity,
            "change_in_density" : change_in_density,
            "appears_later" : appears_later,
            "disappears" : disappears,
            "merge" : merge,
        }
        self.clusters.append(cluster)

    def generate_clusters(self, X):
        data = []
        for cluster in self.clusters:
            center = cluster["center"]
            velocity = cluster["velocity"]
            num_points = self.num_points_per_cluster
            cluster_data = self.generate_cluster_data(center, velocity, num_points)
            data.extend(cluster_data)



            # Generate anomalies
        anomalies = np.random.rand(self.num_point_anom, self.num_dim)
        for i in range(self.num_dim):
            anomalies[:, i] = anomalies[:, i] * (max(X[:, i]) - min(X[:, i])) + min(X[:, i])
        return data

    def generate_cluster_data(self, center, velocity, num_points):
        data = []
        for _ in range(num_points):
            point = center + np.random.normal(0, 0.1, 2)  # Add some random noise
            data.append(point)
            center += velocity  # Move the center according to the velocity
        return data
    
    def get_instance():
        pass



