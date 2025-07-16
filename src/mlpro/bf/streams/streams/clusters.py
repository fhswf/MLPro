## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - The integrative middleware framework for standardized machine learning
## -- Package : mlpro.bf.streams.streams
## -- Module  : clusters.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2023-11-20  0.0.0     SK       Creation
## -- 2024-04-18  1.0.0     SK       First draft implementation
## -- 2024-04-23  1.1.0     SK       Bug fixes
## -- 2024-05-22  1.1.1     SK       Bug fix
## -- 2024-06-04  1.1.2     DA       Bugfix: ESpace instead of MSpace
## -- 2024-06-04  1.2.0     SK       Addition of split and merge functionalities to clusters
## -- 2024-06-16  1.2.1     SK       Optimization and restructuring
## -- 2024-06-17  1.3.0     SK       Functionality for appearance of outliers
## -- 2025-04-02  1.3.1     DA       Little refactoring
## -- 2025-07-16  1.3.2     DA       Little refactoring
## -------------------------------------------------------------------------------------------------

"""
Ver. 1.3.2 (2025-07-16)

This module provides the native stream class StreamMLProClusterGenerator.
These stream provides instances with self._num_dim dimensional random feature data, placed around
self._num_clusters number of centers (random). The resulting clusters or clusters may or may not move,
change size, change density and/or change distribution_bias over time.

"""

import random
import math

import numpy as np

from mlpro.bf.various import Log
from mlpro.bf.math import Element, MSpace, ESpace
from mlpro.bf.streams.basics import *
from mlpro.bf.streams.streams.provider_mlpro import StreamMLProBase



# Export list for public API
__all__ = [ 'StreamMLProClusterGenerator' ]




## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class StreamMLProClusterGenerator (StreamMLProBase):
    """
    This benchmark stream class generates freely configurable random point clusters of any number,
    dimensionality,  size, velocity, acceleration and distribution_bias.

    Parameters
    ----------
    p_num_dim : int
        The number of dimensions or features of the data. Default = 2.
    p_num_instances : int
        Total number of instances. The value '0' means indefinite. Default = 1000.
    p_num_clusters : int
        Number of clusters. Default = 4.
    p_outlier_appearance : bool
        If there are outliers. Default = False.
    p_outlier_rate : float
        The rate at which outliers occur. Default = 0.05.
    p_radii : list
        Radii of the clusters. Default = 100.
    p_velocities : list
        Velocity of the clusters in unit 1/instance. Default = 0.0.
    p_distribution_bias : list[]
        List of integer distribution_bias. For example, a list [1,2] causes the first cluster 
        to be flooded with two times more instances than the second one. If empty , all clusters 
        are flooded equally.
    p_change_radii : bool
        If there are changes in radii of clusters. Default= False.
    p_rate_of_change_of_radius : float
        Rate of change of radius per cycle. Default = 0.001.
    p_points_of_change_radii : list
        Instances at which change of radii occur. Default = None.
    p_num_clusters_for_change_radii : list
        The number of clusters for which the radius changes. Default = None.
    p_change_velocities : bool
        If there are changes in velocities of clusters. Default = False.
    p_points_of_change_velocities : list
        Instances at which change of velocities occur. Default = None.
    p_num_clusters_for_change_velocities : list
        The number of clusters for which the velocity changes. Default = None.
    p_changed_velocities : list
        The changed value of velocities. Default = None.
    p_change_distribution_bias : bool
        If there are changes in distribution_biass of clusters. Default = False.
    p_points_of_change_distribution_bias : list
        Instances at which change of distribution_biass occur. Default = None.
    p_num_clusters_for_change_distribution_bias : list
        The number of clusters for which the radius changes. Default = None.
    p_appearance_of_clusters : bool
        If new clusters appear. Default = False.
    p_points_of_appearance_of_clusters : list
        Instances at which new clusters appears. Deafult = None.
    p_num_new_clusters_to_appear : int
        The number of clusters to appear. Default = None.
    p_disappearance_of_clusters : bool
        If clusters disappears. Default = False.
    p_points_of_disappearance_of_clusters : list
        Instances at which clusters disappear. Default = None.
    p_num_clusters_to_disappear : int
        The number of clusters which disappear in time. Default = None.
    p_split_and_merge_of_clusters : bool
        If clusters split and merge. Default = False.
    p_num_of_clusters_for_split_and_merge : int
        Number of clusters that split split at first and then later on merge. Default = 2.
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
    C_VERSION               = '1.2.1'
    C_SCIREF_ABSTRACT       = 'Demo stream provides self.C_NUM_INSTANCES self._no_dim-dimensional instances randomly positioned around centers which form clusters whose numbers, size, velocity, acceleration and density may or may not change over time.'
    C_BOUNDARIES            = [-1000,1000]

## -------------------------------------------------------------------------------------------------
    def __init__( self,
                  p_num_dim : int = 2,
                  p_num_instances : int = 1000,
                  p_num_clusters : int = 4,
                  p_outlier_appearance : bool = False,
                  p_outlier_rate : float = 0.05,
                  p_radii : list = [100.0],
                  p_velocities : list = [0.0],
                  p_distribution_bias : list = [1],
                  p_change_radii : bool = False,
                  p_rate_of_change_of_radius : float = 0.001,
                  p_points_of_change_radii : list = None,
                  p_num_clusters_for_change_radii : int = None,
                  p_change_velocities : bool = False,
                  p_points_of_change_velocities : list = None,
                  p_num_clusters_for_change_velocities : int = None,
                  p_changed_velocities : list = None,
                  p_change_distribution_bias : bool = False,
                  p_points_of_change_distribution_bias : list = None,
                  p_num_clusters_for_change_distribution_bias : int = None,
                  p_appearance_of_clusters : bool = False,
                  p_points_of_appearance_of_clusters : list = None,
                  p_num_new_clusters_to_appear : int = None,
                  p_disappearance_of_clusters : bool = False,
                  p_points_of_disappearance_of_clusters : list = None,
                  p_num_clusters_to_disappear : int = None,
                  p_clusters_split_and_merge : bool = False,
                  p_clusters_split : bool = False,
                  p_points_of_split : list = None,
                  p_velocities_after_split : list = None,
                  p_num_clusters_to_split_into: int = 2,
                  p_max_clusters_affected : float = 0.75,
                  p_seed = None,
                  p_logging = Log.C_LOG_ALL,
                  **p_kwargs ):
        
        if p_seed is not None:
            self.set_random_seed(p_seed=p_seed)
        else:
            random.seed()
            np.random.seed()
        
        # Initialize parameters        
        self._num_dim               = p_num_dim
        self._num_clusters          = p_num_clusters
        self._radii                 = self._extend_property_list(p_radii, self._num_clusters)
        self._velocities            = self._extend_property_list(p_velocities, self._num_clusters)
        self._distribution_bias     = self._extend_property_list(p_distribution_bias, self._num_clusters)
        self._clusters              = {}
        self._cluster_ids           = [x + 1 for x in range(self._num_clusters)]
        self._current_cluster       = 1
        self._cycle                 = 1
        self.C_NUM_INSTANCES        = p_num_instances
        self._change_in_radius      = self._change_in_property(change=p_change_radii,
                                                               point_of_change=p_points_of_change_radii,
                                                               num_clusters_affected=p_num_clusters_for_change_radii)
        self._roco_radius           = p_rate_of_change_of_radius
        self._change_in_velocity    = self._change_in_property(change=p_change_velocities,
                                                               point_of_change=p_points_of_change_velocities,
                                                               num_clusters_affected=p_num_clusters_for_change_velocities,
                                                               changed_values=p_changed_velocities)
        self._change_in_distribution_bias= self._change_in_property(change=p_change_distribution_bias,
                                                               point_of_change=p_points_of_change_distribution_bias,
                                                               num_clusters_affected=p_num_clusters_for_change_distribution_bias)
        self._new_cluster           = self._change_in_property(change=p_appearance_of_clusters,
                                                               point_of_change=p_points_of_appearance_of_clusters,
                                                               num_clusters_affected=p_num_new_clusters_to_appear)
        self._remove_cluster        = self._change_in_property(change=p_disappearance_of_clusters,
                                                               point_of_change=p_points_of_disappearance_of_clusters,
                                                               num_clusters_affected=p_num_clusters_to_disappear)
        if p_clusters_split:
            self._split_and_merge_of_clusters= self._change_in_property(change=p_clusters_split,
                                                                        point_of_change=p_points_of_split,
                                                                        num_clusters_affected=p_num_clusters_to_split_into)
        else:
            self._split_and_merge_of_clusters= self._change_in_property(change=p_clusters_split_and_merge,
                                                                        point_of_change=p_points_of_split,
                                                                        num_clusters_affected=p_num_clusters_to_split_into)
        self._split_of_clusters = p_clusters_split

        if self._split_and_merge_of_clusters:
            self._split_and_merge_of_clusters["start"] = self._split_and_merge_of_clusters["start"][:1]
            self._split_and_merge_of_clusters["end"] = [self.C_NUM_INSTANCES*0.8]
            self._velocities_split = []
            self._center_split = []
            if p_velocities_after_split:
                if len(p_velocities_after_split) == 1:
                    self._velocities_split = p_num_clusters_to_split_into*p_num_clusters_to_split_into
                elif len(p_velocities_after_split) < p_num_clusters_to_split_into:
                    self._velocities_split = p_velocities_after_split.extend([p_velocities_after_split[0]] *
                                                    (p_num_clusters_to_split_into-len(p_velocities_after_split)))
                else:
                    self._velocities_split = p_velocities_after_split[:p_num_clusters_to_split_into]

            else:
                for x in range(p_num_clusters_to_split_into):
                    if max(self._velocities)==0 and min(self._velocities)==0:
                        self._velocities_split.append(random.random())
                    else:
                        a = random.randint(0,1)
                        if a == 0:
                            self._velocities_split.append(max(self._velocities)*random.random())
                        else:
                            self._velocities_split.append(max(self._velocities)*(1+random.random()))
        
        self._max_clusters_affected = p_max_clusters_affected

        self._outlier_appearance = p_outlier_appearance
        self._outlier_rate       = p_outlier_rate
    
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
    def _change_in_property(self, change = False, point_of_change = None, num_clusters_affected = None,
                            changed_values = None):
        """
        Function to assign clusters and the point of change, when there is a change in property
        """
        if not change:
            return False
        
        if num_clusters_affected:
            num_clusters = num_clusters_affected
        else:
            num_clusters = random.randint(1, int(self._max_clusters_affected*self._num_clusters))

        clusters = list(random.sample(self._cluster_ids, num_clusters))

        if point_of_change is None:           
            start = [random.randint(int(0.1*self.C_NUM_INSTANCES), int(0.5*self.C_NUM_INSTANCES))
                     for _ in clusters]
        else:
            start = point_of_change
            if len(start) == 1:
                start *= num_clusters_affected
            elif len(start) < num_clusters_affected:
                start.extend([start[0]] * (num_clusters_affected - len(start)))
            elif len(start) >= num_clusters_affected:
                start = start[:num_clusters_affected]



        y = 200 if self.C_NUM_INSTANCES > 1000 else int(0.2 * self.C_NUM_INSTANCES)
        z = [x+y for x in start]
        end = [random.randint(x, self.C_NUM_INSTANCES) for x in z]

        if changed_values:
            if len(changed_values) == 1:
                changed_values = changed_values*num_clusters

        return {"clusters": clusters, "start": start, "end": end, "changed_values": changed_values}


## -------------------------------------------------------------------------------------------------
    def _setup_feature_space(self) -> MSpace:
        feature_space : MSpace = ESpace()

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
        for cluster_id in self._cluster_ids:
            # Add cluster to dictionary
            self._clusters[cluster_id] = self._define_cluster(cluster_id)


## -------------------------------------------------------------------------------------------------
    def _define_cluster(self, cluster_id):
        """
        Function to define the cluster.
        """
        center = np.array([random.randint(self.C_BOUNDARIES[0], self.C_BOUNDARIES[1])
                           for _ in range(self._num_dim)])
        velocity = self._find_velocity(self._velocities[cluster_id - 1])
        return {
            "center": center,
            "radius": self._radii[cluster_id - 1],
            "velocity": velocity,
            "roc_of_radius": 0.0,
            "distribution_bias": self._distribution_bias[cluster_id - 1]
        }


## -------------------------------------------------------------------------------------------------
    def _find_velocity(self, velocity, init_point : list = [], final_point : list = [], steps : int = 1):
        """
        Function to calculate velocity of a cluster.
        """
        if len(final_point) != 0:
            return [(float(final_point[d]) - init_point[d]) / steps for d in range(self._num_dim)]

        velocity_vector = np.array([random.uniform(self.C_BOUNDARIES[0], self.C_BOUNDARIES[1])
                                    for _ in range(self._num_dim)])
        dist = np.linalg.norm(velocity_vector)
        return velocity_vector * (velocity / dist)
    

## -------------------------------------------------------------------------------------------------
    def _get_next(self) -> Instance:
        if self.C_NUM_INSTANCES== 0: pass
        elif self._index == self.C_NUM_INSTANCES: raise StopIteration

        self._prepare_clusters_for_changes()
        self._update_cluster_properties()
        cluster_id = self._get_next_cluster()

        feature_data = Element(self._feature_space)

        if self._outlier_appearance and (random.uniform(0, 1) <= self._outlier_rate):
            point_values = np.array([random.randint(self.C_BOUNDARIES[0], self.C_BOUNDARIES[1])
                                     for _ in range(self._num_dim)])
        else:
            point_values = self._generate_random_point_around_cluster(cluster_id)
        
        feature_data.set_values(point_values)

        self._index += 1
        return Instance(p_feature_data=feature_data)
    

## -------------------------------------------------------------------------------------------------
    def _prepare_clusters_for_changes(self):
        if self._change_in_radius:
            self._handle_change_in_property("radius", self._change_in_radius, self._roco_radius)

        if not self._split_and_merge_of_clusters:
            if self._change_in_velocity:
                self._handle_change_in_property("velocity", self._change_in_velocity, 0)
        
        if self._change_in_distribution_bias:
            self._handle_change_in_property("distribution_bias", self._change_in_distribution_bias, 0)

        if self._new_cluster:
            self._handle_new_clusters()

        if self._remove_cluster:
            self._handle_remove_clusters()

        if self._split_and_merge_of_clusters:
            self._handle_split_and_merge()
    

## -------------------------------------------------------------------------------------------------
    def _handle_change_in_property(self, property_name, property_change, change_value):
        if self._index in property_change["start"]:
            ids = [i for i, x in enumerate(property_change["start"]) if x == self._index]

            for idx in ids:
                cluster_id = property_change["clusters"][idx]

                if property_name == "radius":
                    self._clusters[cluster_id]["roc_of_radius"] = self._clusters[cluster_id]["radius"] * change_value

                elif property_name == "velocity":
                    if property_change["changed_values"]:
                        self._clusters[cluster_id]["velocity"] = self._find_velocity(velocity=property_change["changed_values"][idx])
                    else:
                        if max(self._velocities)==0 and min(self._velocities)==0:
                            velocity = random.random()*0.25
                        else:
                            a = random.randint(0,1)
                            if a == 0:
                                velocity = max(self._velocities)*random.random()
                            else:
                                velocity = max(self._velocities)*(1+random.random())
                        self._clusters[cluster_id]["velocity"] = self._find_velocity(velocity=velocity)

                elif property_name == "distribution_bias":
                    if max(self._distribution_bias)==1 and min(self._distribution_bias)==1:
                        distribution_bias = random.randint(1,10)
                    else:
                        distribution_bias = random.randint(1,max(self._distribution_bias)+2)
                    self._clusters[cluster_id]["distribution_bias"] = distribution_bias

        if self._index in property_change["end"]:
            ids = [i for i, x in enumerate(property_change["end"]) if x == self._index]
            for idx in ids:
                cluster_id = property_change["clusters"][idx]
                if property_name == "radius":
                    self._clusters[cluster_id]["roc_of_radius"] = 0


## -------------------------------------------------------------------------------------------------
    def _handle_new_clusters(self):
        if self._index in self._new_cluster["start"]:
            n = self._new_cluster["start"].count(self._index)
            for _ in range(n):
                self._num_clusters += 1
                self._cluster_ids.append(self._num_clusters)
                self._radii.append((min(self._radii) + max(self._radii))/2)
                a = random.randint(0, 1)
                self._velocities.append(min(self._velocities) if a == 0 else (min(self._velocities) + max(self._velocities)) / 2)
                self._distribution_bias.append(random.randint(1, max(self._distribution_bias)))
                self._clusters[self._cluster_ids[-1]] = self._define_cluster(cluster_id=self._cluster_ids[-1])


## -------------------------------------------------------------------------------------------------
    def _handle_remove_clusters(self):
        if self._index in self._remove_cluster["start"]:
                ids = [i for i, x in enumerate(self._remove_cluster["start"]) if x == self._index]
                j=1
                for i in ids:
                    self._radii.pop(i-j)
                    self._velocities.pop(i-j)
                    self._distribution_bias.pop(i-j)
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


## -------------------------------------------------------------------------------------------------
    def _handle_split_and_merge(self):
        if self._index < 2:
            self._center_split = self._clusters[self._split_and_merge_of_clusters["clusters"][0]]["center"]
            for id in self._split_and_merge_of_clusters["clusters"]:
                self._clusters[id]["velocity"] = np.zeros(self._num_dim)
                self._clusters[id]["center"] = self._center_split

        if self._index in self._split_and_merge_of_clusters["start"]:
            x = 0
            for id in self._split_and_merge_of_clusters["clusters"]:
                self._clusters[id]["velocity"] = self._find_velocity(velocity=self._velocities_split[x])
                x += 1

        if not self._split_of_clusters:

            if self._index == int(self._split_and_merge_of_clusters["start"][0]+
                                  (self._split_and_merge_of_clusters["end"][0] -
                                  self._split_and_merge_of_clusters["start"][0])/2):
                for id in self._split_and_merge_of_clusters["clusters"]:
                    initial_point = list(self._clusters[id]["center"])
                    final_point = self._center_split
                    self._clusters[id]["velocity"] = self._find_velocity(velocity=0,
                                                                     init_point=initial_point,
                                                                     final_point=final_point,
                                                                     steps=int((self._split_and_merge_of_clusters["end"][0] -
                                                                                self._split_and_merge_of_clusters["start"][0])/2)-1)

            if self._index in self._split_and_merge_of_clusters["end"]:
                for id in self._split_and_merge_of_clusters["clusters"]:
                    self._clusters[id]["velocity"] = np.zeros(self._num_dim)
                    self._clusters[id]["center"] = self._center_split


## -------------------------------------------------------------------------------------------------
    def _update_cluster_properties(self):
        for cluster in self._clusters.values():
            if cluster["roc_of_radius"] != 0:
                cluster["radius"] += cluster["roc_of_radius"]

            cluster["center"] = cluster["center"].astype(float) + cluster["velocity"]


## -------------------------------------------------------------------------------------------------
    def _get_next_cluster(self):
        """
        Function to determine the cluster for the next data point to be added.
        """
        _flag = False

        for i in range(self._current_cluster- 1, self._num_clusters):

            if (self._cycle % self._clusters[self._current_cluster]["distribution_bias"]) == 0:
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

                if (self._cycle % self._clusters[self._current_cluster]["distribution_bias"]) == 0:
                    c = i + 1
                    self._current_cluster= i + 2
                    break
                else:
                    self._current_cluster+= 1
        
        return c


## -------------------------------------------------------------------------------------------------
    def _generate_random_point_around_cluster(self, cluster_id):
        cluster = self._clusters[cluster_id]
        center = cluster["center"]
        radius = cluster["radius"]
        point_values = np.zeros(self._num_dim)

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
                point_values[d] = center[d] + random.randint(0, int(2 * radius)) - radius

        return point_values
    
