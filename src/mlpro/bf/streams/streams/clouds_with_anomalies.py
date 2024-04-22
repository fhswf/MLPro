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
                  p_change_in_radius : bool = False,
                  p_change_in_velocity : bool = False,
                  p_change_in_weight : bool = False,
                  p_appearance_of_new_cluster : bool = False,
                  p_disappearance_of_cluster : bool = False,
                  p_seed = None,
                  p_logging = Log.C_LOG_ALL,
                  **p_kwargs ):
        
        self._num_dim            = p_num_dim
        self._num_clouds         = int(p_num_clouds)
        self._radii              = p_radii
        self._velocity           = p_velocity
        self._weight             = p_weight
        self._cloud_ids          = []
        self._cloud_id           = 1
        self._clouds             = {}
        self._cycle              = 1
        self.C_NUM_INSTANCES     = p_num_instances
        self._change_in_radius   = p_change_in_radius
        self._change_in_velocity = p_change_in_velocity
        self._change_in_weight   = p_change_in_weight
        self._new_cluster        = p_appearance_of_new_cluster
        self._remove_cluster     = p_disappearance_of_cluster

        self.set_random_seed(p_seed=p_seed)

        for x in range(self._num_clouds):
            self._cloud_ids.append(x+1)

        self._clouds_affected    = []
        for _ in range(5):
            self._clouds_affected.append(random.sample(self._cloud_ids, random.randint(1,int(0.8*self._num_clouds))))

        self._point_of_change    = [[random.randint(0.1*self.C_NUM_INSTANCES, 0.4*self.C_NUM_INSTANCES) for _ in range(5)],
                                   [random.randint(0.6*self.C_NUM_INSTANCES, self.C_NUM_INSTANCES) for _ in range(5)]]

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

        for a in self._cloud_ids:

            # Add cloud to dictionary
            self._clouds[str(a)] = self._add_cloud(a)


## -------------------------------------------------------------------------------------------------
    def _add_cloud(self, id):

        cloud = {}
        
        # 1 Compute the initial position of the center of the cloud
        center = np.zeros(self._num_dim)

        for b in range(self._num_dim):
            center[b] = random.randint(self.C_BOUNDARIES[0], self.C_BOUNDARIES[1])

        cloud["center"] = center

        # 2 Assign the radius to the cloud
        if len(self._radii) == 1:
            cloud["radius"] = self._radii[0]

        elif len(self._radii) == self._num_clouds:
            cloud["radius"] = self._radii[id-1]

        elif len(self._radii) == self._num_clouds-1:
            self._radii.append((min(self._radii) + max(self._radii))/2)
            cloud["radius"] = self._radii[id-1]

        # 3 Assign velocity to the cloud
        if len(self._velocity) == 1:
            velocity = self._velocity[0]

        elif len(self._velocity) == self._num_clouds:
            velocity = self._velocity[id-1]

        elif len(self._velocity) == self._num_clouds-1:
            a = random.randint(0, 1)
            self._velocity.append(min(self._velocity) if a == 0 else (min(self._velocity) + max(self._velocity)) / 2)
            velocity = self._velocity[id-1]

        cloud["velocity"] = self._find_velocity(velocity=velocity)

        # 5 Assign rate of change of radius as zero
        cloud["roc_of_radius"] = 0.0

        # 6 Assign weight to the cloud
        if len(self._weight) == 1:
            cloud["weight"] = self._weight[0]

        elif len(self._weight) == self._num_clouds:
            cloud["weight"] = self._weight[id-1]

        elif len(self._weight) == self._num_clouds-1:
            self._weight.append(random.randint(min(self._weight), max(self._weight)))
            cloud["weight"] = self._weight[id-1]

        return cloud


## -------------------------------------------------------------------------------------------------
    def _find_velocity(self, velocity):

        velocity_vector = np.zeros(self._num_dim)
        dist = 0

        for d in range(self._num_dim):
            velocity_vector[d] = random.randint(self.C_BOUNDARIES[0], self.C_BOUNDARIES[1])
            dist = dist + velocity_vector[d]**2

        dist = dist**0.5
        f    = velocity / dist

        for d in range(self._num_dim):
            velocity_vector[d] *= f

        return velocity_vector


## -------------------------------------------------------------------------------------------------
    def _get_next(self) -> Instance:

        # 0 Preparation
        if self.C_NUM_INSTANCES== 0: pass
        elif self._index == self.C_NUM_INSTANCES: raise StopIteration

        
        if self._index == self._point_of_change[0][0]:
            if self._change_in_radius:
                for id in self._clouds_affected[0]:
                    self._clouds[str(id)]["roc_of_radius"] = self._clouds[str(id)]["radius"]*0.001
        if self._index == self._point_of_change[1][0]:
            if self._change_in_radius:
                for id in self._clouds_affected[0]:
                    self._clouds[str(id)]["roc_of_radius"] = 0.0

        if self._index == self._point_of_change[0][1]:
            if self._change_in_velocity:
                for id in self._clouds_affected[1]:
                    a = random.randint(0,2)
                    if a == 0:
                        velocity = 0
                    else:
                        if max(self._velocity)==0 and min(self._velocity)==0:
                            velocity = random.random()
                        else:
                            velocity = max(self._velocity)*random.random()
                    self._clouds[str(id)]["velocity"] = self._find_velocity(velocity)

        if self._index == self._point_of_change[0][2]:
            if self._change_in_weight:
                for id in self._clouds_affected[2]:
                    if max(self._weight)==1 and min(self._weight)==1:
                        weight = random.randint(1,10)
                    else:
                        weight = random.randint(1,max(self._weight)+2)
                    self._clouds[str(id)]["weight"] = weight

        if self._index == self._point_of_change[0][3]:
            if self._new_cluster:
                self._num_clouds += 1
                self._cloud_ids.append(self._num_clouds)
                self._clouds[str(self._cloud_ids[-1])] = self._add_cloud(id=self._cloud_ids[-1])

        if self._index == self._point_of_change[0][4]:
            if self._remove_cluster:
                if self._num_clouds == 1:
                    raise Exception
                elif self._num_clouds == 2:
                    self._num_clouds -= 1
                    self._cloud_ids.pop(-1)
                    self._cloud_id = 1
                    del self._clouds[str(2)]
                else:
                    a = sorted(random.sample(self._cloud_ids, random.randint(1, int(0.5*self._num_clouds))))
                    if len(self._radii)==self._num_clouds:
                        j=1
                        for i in a:
                            self._radii.pop(i-j)
                            j+=1
                    if len(self._velocity)==self._num_clouds:
                        j=1
                        for i in a:
                            self._velocity.pop(i-j)
                            j+=1
                    if len(self._weight)==self._num_clouds:
                        j=1
                        for i in a:
                            self._weight.pop(i-j)
                            j+=1

                    self._num_clouds -= len(a)
                    self._cloud_ids = self._cloud_ids[:self._num_clouds-len(a)]
                    self._cloud_id = 1
                    for id in a:
                        del self._clouds[str(id)]
                    a = 1
                    clouds = {}
                    for id in self._clouds.keys():
                        clouds[str(a)] = self._clouds[id]
                        a += 1
                    self._clouds = clouds


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

