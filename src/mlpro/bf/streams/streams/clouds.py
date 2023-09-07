## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - A Synoptic Framework for Standardized Machine Learning Tasks
## -- Package : mlpro.bf.streams.streams
## -- Module  : clouds.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2023-08-29  0.0.0     SR       Creation 
## -- 2023-08-29  1.0.0     SR       First draft implementation
## -------------------------------------------------------------------------------------------------

"""
Ver. 1.0.0 (2023-08-29)

This module provides the native stream class StreamMLProClouds. This stream provides instances with
self.C_NUM_DIMENSIONS dimensional random feature data placed around centers (can be defined by user)
which may or maynot move over time.
"""

import numpy as np
import random
from mlpro.bf.streams.models import *
from mlpro.bf.streams.streams.provider_mlpro import StreamMLProBase




## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class StreamMLProClouds (StreamMLProBase):
    """
    This demo stream provides self.C_NUM_INSTANCES 3-dimensional instances randomly positioned around
    centers which move over time.

    p_pattern : str
        Pattern for cloud movements. Possible values are 'random', 'random chain', 'static', 'merge'.
        Default = 'random'.
    p_no_clouds : int
        Number of clouds. Default = 4.
    p_variance : float
        Variance of points around the cloud centeres. Default = 5.0.
    p_velocity : float
        Velocity factor for the centers. Default = 0.1.
    p_logging
        Log level (see constants of class Log). Default: Log.C_LOG_ALL.
    """

    C_ID                    = 'DynamicClouds3D'
    C_NAME                  = 'Dynamic Clouds 3D'
    C_TYPE                  = 'Demo'
    C_VERSION               = '1.0.0'
    C_NUM_DIMENSIONS        = 3
    C_NUM_INSTANCES         = 0
    C_SCIREF_ABSTRACT       = 'Demo stream provides self.C_NUM_INSTANCES C_NUM_DIMENSIONS-dimensional instances per cluster randomly positioned around centers which may or maynot move over time.'
    C_BOUNDARIES            = [-1000,1000]

## -------------------------------------------------------------------------------------------------
    def __init__( self,
                  p_num_dim : int = 3,
                  p_num_instances : int = 0,
                  p_num_clouds : int = 8,
                  p_radii : list = [100.0],
                  p_behaviour : str = 'static',
                  p_velocity : float = 0.1,
                  p_logging = Log.C_LOG_ALL,
                  **p_kwargs ):
        
        StreamMLProBase.__init__(self,
                                 p_logging=p_logging,
                                 **p_kwargs)
        
        if str.lower(p_pattern) not in self.C_PATTERN:
            raise ValueError(f"Invalid value for pattern, allowed values are {self.C_PATTERN}")
        
        self.num_dim = p_num_dim
        self.radii = p_radii
        self.num_clouds = int(p_num_clouds)
        self.C_NUM_INSTANCES = self.C_NUM_INST_PER_CLOUD*self.num_clouds
        self.velocity = p_velocity
        self.cloud_centers = []
        self.num_instances = p_num_instances
        self.behaviour = str.lower(p_behaviour)
        self.centers_step = []


## -------------------------------------------------------------------------------------------------
    def _setup_feature_space(self) -> MSpace:
        feature_space : MSpace = MSpace()

        for i in range(3):
            feature_space.add_dim( Feature( p_name_short = 'f' + str(i),
                                            p_base_set = Feature.C_BASE_SET_R,
                                            p_name_long = 'Feature #' + str(i),
                                            p_name_latex = '',
                                            # p_boundaries = self.C_BOUNDARIES,
                                            p_description = '',
                                            p_symmetrical = False,
                                            p_logging=Log.C_LOG_NOTHING ) )

        return feature_space


## -------------------------------------------------------------------------------------------------
    def _init_dataset(self):

        # 1 Preparation
        try:
            seed = Stream.set_random_seed(p_seed=32)
        except:
            seed = random.seed(32)

        self._dataset = np.empty( (self.C_NUM_INSTANCES, self.num_dim))


        # Compute the initial positions of the centers
        for i in range(self.num_clouds):

            self.cloud_centers.append([])

            for j in range(self.num_dim):
                self.cloud_centers[i].append(random.randint(self.C_BOUNDARIES[0],
                                                           self.C_BOUNDARIES[1]))

       
        if self.behaviour == 'dynamic':

            # Compute the final positions of the centers
            for i in range(self.num_clouds):

                self.centers_step.append([])

                for j in range(self.num_dim):
                    self.centers_step[i].append(random.randint(self.C_BOUNDARIES[0],
                                                           self.C_BOUNDARIES[1]))

            for x in range(self.num_clouds):

                mag = 0
                for i in range(self.num_dim):
                    mag = mag + (self.cloud_centers[x][i]-self.centers_step[x][0])**2
                mag = mag**0.5
                if mag != 0:
                    self.centers_step[x][:] = ((self.centers_step[x][:]-self.cloud_centers[x][:])/ mag)*self.velocity
                else:
                    self.centers_step[x][:] = (1/(self.num_dim**0.5))*self.velocity


        


    def _get_next(self) -> Instance:

        if self._index == self.C_NUM_INSTANCES: raise StopIteration

        if self.behaviour == 'dynamic':
            self.cloud_centers = self.cloud_centers + self.centers_step

        id = random.randint(0, self.num_clouds)
        if len(self.radii) == 1:
            radius = self.radii[0]
        else:
            radius = self.radii[id]

        instance = []

        for i in range(self.num_dim):
            instance.append(random.random()*radius)
            a = random.random()
            if a<0.5:
                instance[i] = instance[i]*(-1)

        instance = instance + self.cloud_centers[id]


        self._index += 1

        return instance


