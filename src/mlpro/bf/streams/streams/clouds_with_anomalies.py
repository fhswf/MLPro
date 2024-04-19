## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - A Synoptic Framework for Standardized Machine Learning Tasks
## -- Package : mlpro.bf.streams.streams
## -- Module  : cluster_generator.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2023-11-20  0.0.0     SK       Creation
## -- 2024-04-18  1.0.0     SK       First draft implementation
## -------------------------------------------------------------------------------------------------

"""
Ver. 1.0.0 (2024-04-18)

This module provides the native stream class StreamMLProClusterbasedAnomalies.
These stream provides instances with self._num_dim dimensional random feature data, placed around
self._num_clouds number of centers (random). The resulting clouds or clusters may or may not move,
change size, change density and/or change weight over time.

"""

import random
import math
from mlpro.bf.streams.models import *
from mlpro.bf.streams.streams.provider_mlpro import StreamMLProBase



## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class StreamMLProClusterbasedAnomalies (StreamMLProBase):
    """
    This benchmark stream class generates freely configurable random point clouds of any number,
    dimensionality,  size, velocity, acceleration and weight.

    Parameters
    ----------
    p_num_dim : int
        The number of dimensions or features of the data. Default = 2.
    p_num_instances : int
        Total number of instances. The value '0' means indefinite. Default = 1000.
    p_num_clouds : int
        Number of clouds. Default = 4.
    p_radii : list
        Radii of the clouds. Default = 100.
    p_velocity : list
        Velocity of the clouds in unit 1/instance. Default = 0.0.
    p_weights : list[]
        Optional list of integer weights per cloud. For example, a list [1,2] causes the first cloud 
        to be flooded with two times more instances than the second one. If empty , all clouds 
        are flooded equally.
    p_seed 
        Seeding value for the random generator. Default = None (no seeding).
    p_logging
        Log level (see constants of class Log). Default: Log.C_LOG_ALL.
    """

    C_ID                    = 'ClusterbasedAnomalies'
    C_NAME                  = 'Clouds N-Dim'
    C_TYPE                  = 'Benchmark'
    C_VERSION               = '1.0.0'
    C_SCIREF_ABSTRACT       = 'Demo stream provides self.C_NUM_INSTANCES self._no_dim-dimensional instances randomly positioned around centers which form clusters whose numbers, size, velocity, acceleration and density may or may not change over time.'
    C_BOUNDARIES            = [-1000,1000]

## -------------------------------------------------------------------------------------------------
    def __init__( self,
                  p_num_dim : int = 2,
                  p_num_instances : int = 1000,
                  p_num_clouds : int = 4,
                  p_radii : list = [100.0],
                  p_velocity : list = [0.0],
                  p_weight : list = [1],
                  p_seed = None,
                  p_logging = Log.C_LOG_ALL,
                  **p_kwargs ):
        
        self._num_dim         = p_num_dim
        self._radii           = p_radii
        self._num_clouds      = int(p_num_clouds)
        self._velocity        = p_velocity
        self._weight          = p_weight
        self._cloud_ids       = []
        self._cloud_id        = 1
        self._clouds          = {}
        self._cycle           = 1

        self.C_NUM_INSTANCES  = p_num_instances

        self.set_random_seed(p_seed=p_seed)

        for x in range(self._num_clouds):
            self._cloud_ids.append(x+1)

        self._no_cloud_ids = len(self._cloud_ids)

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


        """cluster = {
            "change_in_radius" : change_in_radius = 0.1,
            "change_in_velocity" : change_in_velocity,
            "change_in_density" : change_in_density,
            "appears_later" : appears_later,
            "disappears" : disappears,
            "merge" : merge,
        }
        self.clusters.append(cluster)"""

        for a in range(self._num_clouds):

            cloud = {}
            
            # 1 Compute the initial position of the center of the cloud
            center = np.zeros(self._num_dim)

            for b in range(self._num_dim):
                center[b] = random.randint(self.C_BOUNDARIES[0], self.C_BOUNDARIES[1])

            cloud["center"] = center

            # 2 Assign the radius to the cloud
            if len(self._radii) == 1:
                cloud["radius"] = self._radii[0]

            else:
                cloud["radius"] = self._radii[a]


            # 3 Assign velocity to the cloud
            if len(self._velocity) == 1:
                velocity = self._velocity[0]

            else:
                velocity = self._velocity[a]

            velocity_vector = np.zeros(self._num_dim)
            dist = 0

            for d in range(self._num_dim):
                velocity_vector[d] = random.randint(self.C_BOUNDARIES[0], self.C_BOUNDARIES[1])
                dist = dist + velocity_vector[d]**2

            dist = dist**0.5
            f    = velocity / dist

            for d in range(self._num_dim):
                velocity_vector[d] *= f 
                
            cloud["velocity"] = velocity_vector

            # 5 Assign rate of change of radius as zero
            cloud["roc_of_radius"] = 0.0

            # 6 Assign weight to the cloud
            if len(self._weight) == 1:
                cloud["weight"] = self._weight[0]

            else:
                cloud["weight"] = self._weight[a]

            # 7 Add cloud to dictionary
            self._clouds[str(a+1)] = cloud


## -------------------------------------------------------------------------------------------------
    def _get_next(self) -> Instance:

        # 0 Preparation
        if self.C_NUM_INSTANCES== 0: pass
        elif self._index == self.C_NUM_INSTANCES: raise StopIteration


        # 1 Update of center positions
        for a in self._cloud_ids:
            self._clouds[str(a)]["center"] = self._clouds[str(a)]["center"] + self._clouds[str(a)]["velocity"]


        # 2 Selection of the cloud to be fed
        _flag = False

        for i in range(self._cloud_id - 1, self._num_clouds):

            if (self._cycle % self._clouds[str(self._cloud_id)]["weight"]) == 0:
                c = i + 1
                self._cloud_id = i + 2
                _flag = True
                break
            else:
                self._cloud_id += 1        
        
        if self._cloud_id > self._num_clouds:
            self._cycle += 1
            self._cloud_id = 1

        if _flag == False:
            for i in range(self._cloud_id-1, self._num_clouds):

                if (self._cycle % self._clouds[str(self._cloud_id)]["weight"]) == 0:
                    c = i + 1
                    self._cloud_id = i + 2
                    break
                else:
                    self._cloud_id += 1


        # 3 Generation of random point around the selected cloud
        center       = self._clouds[str(c)]["center"]
        feature_data = Element(self._feature_space)
        point_values = np.zeros(self._num_dim)

        radius = self._clouds[str(c)]["radius"] + self._clouds[str(c)]["roc_of_radius"]

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



