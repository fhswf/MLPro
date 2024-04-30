## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - A Synoptic Framework for Standardized Machine Learning Tasks
## -- Package : mlpro.bf.streams.streams
## -- Module  : clusters.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2023-11-20  0.0.0     SK       Creation
## -- 2024-04-18  1.0.0     SK       First draft implementation
## -- 2024-04-23  1.1.0     SK       Bug fixes
## -------------------------------------------------------------------------------------------------

"""
Ver. 1.0.0 (2024-04-18)

This module provides the native stream class StreamMLProClusterGenerator.
These stream provides instances with self._num_dim dimensional random feature data, placed around
self._num_clusters number of centers (random). The resulting clusters or clusters may or may not move,
change size, change density and/or change weight over time.

"""

import random
import math
from mlpro.bf.streams.models import *
from mlpro.bf.streams.streams.provider_mlpro import StreamMLProBase



## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class StreamMLProClusterGenerator (StreamMLProBase):
    """
    This benchmark stream class generates freely configurable random point clusters of any number,
    dimensionality,  size, velocity, acceleration and weight.

    Parameters
    ----------
    p_num_dim : int
        The number of dimensions or features of the data. Default = 2.
    p_num_instances : int
        Total number of instances. The value '0' means indefinite. Default = 1000.
    p_num_clusters : int
        Number of clusters. Default = 4.
    p_radii : list
        Radii of the clusters. Default = 100.
    p_velocities : list
        Velocity of the clusters in unit 1/instance. Default = 0.0.
    p_weights : list[]
        List of integer weights per cluster. For example, a list [1,2] causes the first cluster 
        to be flooded with two times more instances than the second one. If empty , all clusters 
        are flooded equally.
    p_change_in_radii : bool
        If there are changes in radii of clusters. Default= False.
    p_rate_of_change_of_radius : float
        Rate of change of radius per cycle. Default = 0.001.
    p_points_of_change_in_radii : list
        Instances at which change of radii occur. Default = None.
    p_change_in_velocities : bool
        If there are changes in velocities of clusters. Default= False.
    p_points_of_change_in_velocities : list
        Instances at which change of velocities occur. Default = None.
    p_change_in_weights : bool
        If there are changes in weights of clusters. Default= False.
    p_points_of_change_in_weights : list
        Instances at which change of weights occur. Default = None.
    p_appearance_of_clusters : bool
        If new clusters appear. Default = False.
    p_points_of_appearance_of_clusters : list
        Instances at which new clusters appears. Deafult = None.
    p_disappearance_of_clusters : bool
        If clusters disappears. Default = False.
    p_points_of_disappearance_of_clusters : list
        Instances at which clusters disappear. Default = None.
    p_max_clusters_affected : float
        Fraction of maximum number of clusters affected by changes in properties. Default = 0.75.
    p_seed 
        Seeding value for the random generator. Default = None (no seeding).
    p_logging
        Log level (see constants of class Log). Default: Log.C_LOG_ALL.
    """

    C_ID                    = 'ClustersNDim'
    C_NAME                  = 'Clusters N-Dim'
    C_TYPE                  = 'Benchmark'
    C_VERSION               = '1.0.0'
    C_SCIREF_ABSTRACT       = 'Demo stream provides self.C_NUM_INSTANCES self._no_dim-dimensional instances randomly positioned around centers which form clusters whose numbers, size, velocity, acceleration and density may or may not change over time.'
    C_BOUNDARIES            = [-1000,1000]

## -------------------------------------------------------------------------------------------------
    def __init__( self,
                  p_num_dim : int = 2,
                  p_num_instances : int = 1000,
                  p_num_clusters : int = 4,
                  p_radii : list = [100.0],
                  p_velocities : list = [0.0],
                  p_weights : list = [1],
                  p_change_in_radii : bool = False,
                  p_rate_of_change_of_radius : float = 0.001,
                  p_points_of_change_in_radii : list = None,
                  p_change_in_velocities : bool = False,
                  p_points_of_change_in_velocities : list = None,
                  p_change_in_weights : bool = False,
                  p_points_of_change_in_weights : list = None,
                  p_appearance_of_clusters : bool = False,
                  p_points_of_appearance_of_clusters : list = None,
                  p_disappearance_of_clusters : bool = False,
                  p_points_of_disappearance_of_clusters : list = None,
                  p_max_clusters_affected : float = 0.75,
                  p_seed = None,
                  p_logging = Log.C_LOG_ALL,
                  **p_kwargs ):
        
        # Initialize parameters        
        self._num_dim               = p_num_dim
        self._num_clusters          = p_num_clusters
        self._radii                 = self._extend_property_list(p_radii, self._num_clusters)
        self._velocities              = self._extend_property_list(p_velocities, self._num_clusters)
        self._weights               = self._extend_property_list(p_weights, self._num_clusters)
        self._clusters              = {}
        self._cluster_ids           = [x + 1 for x in range(self._num_clusters)]
        self._current_cluster       = 1
        self._cycle                 = 1
        self.C_NUM_INSTANCES        = p_num_instances
        self._change_in_radius      = self._change_in_property(p_change_in_radii,
                                                               p_points_of_change_in_radii,
                                                               p_max_clusters_affected)
        self._roco_radius           = p_rate_of_change_of_radius
        self._change_in_velocity    = self._change_in_property(p_change_in_velocities,
                                                               p_points_of_change_in_velocities,
                                                               p_max_clusters_affected)
        self._change_in_weight      = self._change_in_property(p_change_in_weights,
                                                               p_points_of_change_in_weights,
                                                               p_max_clusters_affected)
        self._new_cluster           = self._change_in_property(p_appearance_of_clusters,
                                                               p_points_of_appearance_of_clusters,
                                                               p_max_clusters_affected)
        self._remove_cluster        = self._change_in_property(p_disappearance_of_clusters,
                                                               p_points_of_disappearance_of_clusters,
                                                               p_max_clusters_affected)


        self.set_random_seed(p_seed=p_seed)

        StreamMLProBase.__init__ (self,
                                  p_logging=p_logging,
                                  **p_kwargs)
        

## -------------------------------------------------------------------------------------------------
    def _extend_property_list(self, property : list, size : int):
        """
        Function to extend the property list for all clusters.
        """
        if len(property) == 1:
            property *= size

        elif len(property) < size:
            property.extend([property[0]] * (size - len(property)))

        elif len(property) > size:
            property = property[:size]

        return property


## -------------------------------------------------------------------------------------------------
    def _change_in_property(self, change, point_of_change, max_clusters_affected):
        """
        Function to assign clusters and the point of change, when there is a change in property
        """
        if not change:
            return None
        
        dic = {}

        if point_of_change is None:
            num_clusters = random.randint(1,int(max_clusters_affected*self._num_clusters))
            clusters = list(random.sample(self._cluster_ids, num_clusters))
            start = [random.randint(0.1*self.C_NUM_INSTANCES, 0.5*self.C_NUM_INSTANCES) for _ in clusters]
        else:
            clusters = list(random.sample(self._cluster_ids, len(point_of_change)))
            start = point_of_change

        y = 200 if self.C_NUM_INSTANCES > 1000 else 0.2 * self.C_NUM_INSTANCES
        z = [x+y for x in start]
        end = [random.randint(x, self.C_NUM_INSTANCES) for x in z]

        dic["clusters"] = clusters
        dic["start"]    = start
        dic["end"]      = end

        return dic


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
        for a in self._cluster_ids:
            # Add cluster to dictionary
            self._clusters[a] = self._define_cluster(a)


## -------------------------------------------------------------------------------------------------
    def _define_cluster(self, id):
        """
        Function to define the cluster.
        """
        cluster = {}
        
        # 1 Compute the initial position of the center of the cluster
        center = np.zeros(self._num_dim)

        for b in range(self._num_dim):
            center[b] = random.randint(self.C_BOUNDARIES[0], self.C_BOUNDARIES[1])

        cluster["center"] = center

        # 2 Assign the radius to the cluster
        cluster["radius"] = self._radii[id-1]

        # 3 Assign velocity to the cluster
        velocity = self._velocities[id-1]
        cluster["velocity"] = self._find_velocity(velocity=velocity)

        # 5 Assign rate of change of radius as zero initially
        cluster["roc_of_radius"] = 0.0

        # 6 Assign weight to the cluster
        cluster["weight"] = self._weights[id-1]

        return cluster


## -------------------------------------------------------------------------------------------------
    def _find_velocity(self, velocity):
        """
        Fuction to calculate velocity of a cluster.
        """
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

        # 1 Check and prepare clusters for change in properties

        # 1.1 Check and prepare clusters for changes in size
        if self._change_in_radius != None:
            if self._index in self._change_in_radius["start"]:
                ids = [index for index, element in enumerate(self._change_in_radius["start"]) if element == self._index]
                for x in ids:
                    self._clusters[self._change_in_radius["clusters"][x]]["roc_of_radius"] = self._clusters[self._change_in_radius["clusters"][x]]["radius"]*self._roco_radius
            if self._index in self._change_in_radius["end"]:
                ids = [index for index, element in enumerate(self._change_in_radius["end"]) if element == self._index]
                for x in ids:
                    self._clusters[self._change_in_radius["clusters"][x]]["roc_of_radius"] = 0

        # 1.2 Check and prepare clusters for changes in velocity
        if self._change_in_velocity != None:
            if self._index in self._change_in_velocity["start"]:
                ids = [index for index, element in enumerate(self._change_in_velocity["start"]) if element == self._index]
                for x in ids:
                    a = random.randint(0,2)
                    if a == 0:
                        velocity = 0
                    else:
                        if max(self._velocities)==0 and min(self._velocities)==0:
                            velocity = random.random()
                        else:
                            a = random.randint(0,1)
                            if a == 0:
                                velocity = max(self._velocities)*random.random()
                            else:
                                velocity = max(self._velocities)*(1+random.random())
                    self._clusters[self._change_in_velocity["clusters"][x]]["velocity"] = self._find_velocity(velocity)

        # 1.3 Check and prepare clusters for changes in weight
        if self._change_in_weight != None:
            if self._index in self._change_in_weight["start"]:
                ids = [index for index, element in enumerate(self._change_in_weight["start"]) if element == self._index]
                for x in ids:
                    if max(self._weights)==1 and min(self._weights)==1:
                        weight = random.randint(1,10)
                    else:
                        weight = random.randint(1,max(self._weights)+2)
                    self._clusters[self._change_in_weight["clusters"][x]]["weight"] = weight

        # 1.4 Check and prepare for cluster appearances
        if self._new_cluster != None:
            if self._index in self._new_cluster["start"]:
                n = self._new_cluster["start"].count(self._index)
                for _ in range(n):
                    self._num_clusters += 1
                    self._cluster_ids.append(self._num_clusters)
                    self._radii.append((min(self._radii) + max(self._radii))/2)
                    a = random.randint(0, 1)
                    self._velocities.append(min(self._velocities) if a == 0 else (min(self._velocities) + max(self._velocities)) / 2)
                    self._weights.append(random.randint(1, max(self._weight)))
                    self._clusters[self._cluster_ids[-1]] = self._define_cluster(id=self._cluster_ids[-1])

        # 1.5 Check and prepare fro cluster disappearances
        if self._remove_cluster != None:
            if self._index in self._remove_cluster["start"]:
                if self._num_clusters == 1:
                    raise Exception
                elif self._num_clusters == 2:
                    self._num_clusters -= 1
                    self._cluster_ids.pop(self._remove_cluster["cluster"][0]-1)
                    self._current_cluster= 1
                    del self._clusters[self._remove_cluster["cluster"][0]]
                else:
                    ids = [index for index, element in enumerate(self._remove_cluster["start"]) if element == self._index]
                    j=1
                    for i in ids:
                        self._radii.pop(i-j)
                        self._velocities.pop(i-j)
                        self._weights.pop(i-j)
                        j+=1
                    self._num_clusters -= len(ids)
                    self._cluster_ids = self._cluster_ids[:self._num_clusters-len(ids)]
                    self._current_cluster= 1
                    for id in ids:
                        del self._clusters[self._remove_cluster["clusters"][id]]
                        for i in range(len(self._remove_cluster["clusters"])):
                            if self._remove_cluster["clusters"][i] > self._remove_cluster["clusters"][id]:
                                self._remove_cluster["clusters"][i] -= 1
                    a = 1
                    clusters = {}
                    for id in self._clusters.keys():
                        clusters[a] = self._clusters[id]
                        a += 1
                    self._clusters = clusters


        # 2 Update of center positions
        for a in self._cluster_ids:
            self._clusters[a]["center"] = self._clusters[a]["center"] + self._clusters[a]["velocity"]

        # 3 Update of radius
        for a in self._cluster_ids:
            self._clusters[a]["radius"] += self._clusters[a]["roc_of_radius"] 


        # 4 Selection of the cluster to be fed
        c = self._get_next_cluster()


        # 5 Generation of random point around the selected cluster
        center       = self._clusters[c]["center"]
        feature_data = Element(self._feature_space)
        point_values = np.zeros(self._num_dim)

        radius = self._clusters[c]["radius"]

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
    

## -------------------------------------------------------------------------------------------------
    def _get_next_cluster(self):
        """
        Function to determine the cluster for the next data point to be added.
        """
        _flag = False

        for i in range(self._current_cluster- 1, self._num_clusters):

            if (self._cycle % self._clusters[self._current_cluster]["weight"]) == 0:
                c = i + 1
                self._current_cluster= i + 2
                _flag = True
                break
            else:
                self._current_cluster+= 1        

        if self._current_cluster> self._num_clusters:
            self._cycle += 1
            self._current_cluster= 1

        if _flag == False:
            for i in range(self._current_cluster-1, self._num_clusters):

                if (self._cycle % self._clusters[self._current_cluster]["weight"]) == 0:
                    c = i + 1
                    self._current_cluster= i + 2
                    break
                else:
                    self._current_cluster+= 1
        
        return c

