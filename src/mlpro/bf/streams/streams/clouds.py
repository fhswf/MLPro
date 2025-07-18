## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - The integrative middleware framework for standardized machine learning
## -- Package : mlpro.bf.streams.streams
## -- Module  : clouds.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2023-08-29  0.0.0     SK       Creation
## -- 2023-08-29  1.0.0     SK       First draft implementation
## -- 2023-09-15  1.0.1     LSB      Bug Fix
## -- 2023-12-25  1.0.2     DA       Bugfix in StreamMLProClouds._get_next()
## -- 2023-12-26  1.0.3     DA       - Little refactoring of StreamMLProClouds._init_dataset()
## --                                - Bugfix in StreamMLProClouds._get_next()
## -- 2023-12-27  1.1.0     DA       Refactoring
## -- 2023-12-29  1.2.0     DA       Class StreamMLProClouds: new parameter p_weights
## -- 2024-02-06  1.2.1     DA       Class StreamMLProClouds3D8C10000Dynamic: corrections on constants
## -- 2024-02-09  1.2.2     DA       Completion of class documentations
## -- 2024-06-04  1.2.3     DA       Bugfix: ESpace instead of MSpace
## -- 2025-04-02  1.2.4     DA       Little refactoring
## -- 2025-06-06  1.2.5     DA       Little refactoring
## -------------------------------------------------------------------------------------------------

"""
Ver. 1.2.5 (2025-06-06)

This module provides the native stream classes StreamMLProClouds, StreamMLProClouds2D4C1000Static,
StreamMLProClouds3D8C2000Static, StreamMLProClouds2D4C5000Dynamic and StreamMLProClouds3D8C10000Dynamic.
These stream provides instances with self.C_NUM_DIMENSIONS dimensional random feature data, placed around
centers (can be defined by user) which may or maynot move over time.

"""


import random
import math

import numpy as np

from mlpro.bf.various import Log
from mlpro.bf.math import Element, MSpace, ESpace
from mlpro.bf.streams.basics import Feature, Instance
from mlpro.bf.exceptions import ParamError
from mlpro.bf.streams.streams.provider_mlpro import StreamMLProBase



# Export list for public API
__all__ = [ 'StreamMLProClouds',
            'StreamMLProClouds2D4C1000Static',
            'StreamMLProClouds3D8C2000Static',
            'StreamMLProClouds2D4C5000Dynamic',
            'StreamMLProClouds3D8C10000Dynamic' ]




## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class StreamMLProClouds (StreamMLProBase):
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

    C_ID                    = 'CloudsNDim'
    C_NAME                  = 'Clouds N-Dim'
    C_TYPE                  = 'Benchmark'
    C_VERSION               = '1.0.0'
    C_SCIREF_ABSTRACT       = 'Demo stream provides self.C_NUM_INSTANCES C_NUM_DIMENSIONS-dimensional instances per cluster randomly positioned around centers which may or maynot move over time.'
    C_BOUNDARIES            = [-1000,1000]

## -------------------------------------------------------------------------------------------------
    def __init__( self,
                  p_num_dim : int = 3,
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





## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class StreamMLProClouds2D4C1000Static (StreamMLProClouds):
    """
    This benchmark stream generates 1000 2-dimensional instances that form 4 static random point clouds.

    See also: class StreamMLProClouds

    Parameters
    ----------
    p_radii : list
        Radii of the clouds. Default = 20.
    p_seed 
        Seeding value for the random generator. Default = None (no seeding).
    p_logging
        Log level (see constants of class Log). Default: Log.C_LOG_ALL.
    """

    C_ID                    = 'Clouds2D4C1000Static'
    C_NAME                  = 'Static Clouds 2D'
    C_VERSION               = '1.0.1'
    C_NUM_DIMENSIONS        = 2
    C_NUM_INSTANCES         = 1000
    C_SCIREF_ABSTRACT       = 'Demo stream provides 1000 2D instances randomly positioned around four fixed centers.'
    C_BOUNDARIES            = [-100,100]

## -------------------------------------------------------------------------------------------------
    def __init__( self,
                  p_radii : list = [20.0],
                  p_logging = Log.C_LOG_ALL,
                  **p_kwargs ):
        
        StreamMLProClouds.__init__( self,
                                    p_num_dim = self.C_NUM_DIMENSIONS,
                                    p_num_instances = self.C_NUM_INSTANCES,
                                    p_num_clouds = 4,
                                    p_radii = p_radii,
                                    p_velocity = 0.0,
                                    p_logging = p_logging,
                                    **p_kwargs )
            




## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class StreamMLProClouds3D8C2000Static (StreamMLProClouds):
    """
    This benchmark stream generates 2000 3-dimensional instances that form 8 static random point clouds.

    See also: class StreamMLProClouds

    Parameters
    ----------
    p_radii : list
        Radii of the clouds. Default = 20.
    p_seed 
        Seeding value for the random generator. Default = None (no seeding).
    p_logging
        Log level (see constants of class Log). Default: Log.C_LOG_ALL.
    """

    C_ID                    = 'Clouds3D8C2000Static'
    C_NAME                  = 'Static Clouds 3D'
    C_VERSION               = '1.0.1'
    C_NUM_DIMENSIONS        = 3
    C_NUM_INSTANCES         = 2000
    C_SCIREF_ABSTRACT       = 'Demo stream provides 2000 3D instances randomly positioned around eight fixed centers.'
    C_BOUNDARIES            = [-100,100]

## -------------------------------------------------------------------------------------------------
    def __init__( self,
                  p_radii : list = [20.0],
                  p_logging = Log.C_LOG_ALL,
                  **p_kwargs ):
        
        StreamMLProClouds.__init__( self,
                                    p_num_dim = self.C_NUM_DIMENSIONS,
                                    p_num_instances = self.C_NUM_INSTANCES,
                                    p_num_clouds = 8,
                                    p_radii = p_radii,
                                    p_velocity = 0.0,
                                    p_logging = p_logging,
                                    **p_kwargs )





## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class StreamMLProClouds2D4C5000Dynamic (StreamMLProClouds):
    """
    This benchmark stream generates 5000 2-dimensional instances that form 4 dynamic random point clouds.

    See also: class StreamMLProClouds

    Parameters
    ----------
    p_radii : list
        Radii of the clouds. Default = 100.
    p_seed 
        Seeding value for the random generator. Default = None (no seeding).
    p_logging
        Log level (see constants of class Log). Default: Log.C_LOG_ALL.
    """

    C_ID                    = 'Clouds2D4C5000Dynamic'
    C_NAME                  = 'Dynamic Clouds 2D'
    C_VERSION               = '1.0.1'
    C_NUM_DIMENSIONS        = 2
    C_NUM_INSTANCES         = 5000
    C_SCIREF_ABSTRACT       = 'Demo stream provides 2000 2D instances randomly positioned around four randomly moving centers.'
    C_BOUNDARIES            = [-1000,1000]

## -------------------------------------------------------------------------------------------------
    def __init__( self,
                  p_radii : list = [100.0],
                  p_velocity : float = 1,
                  p_logging = Log.C_LOG_ALL,
                  **p_kwargs ):
        
        StreamMLProClouds.__init__( self,
                                    p_num_dim = self.C_NUM_DIMENSIONS,
                                    p_num_instances = self.C_NUM_INSTANCES,
                                    p_num_clouds = 4,
                                    p_radii = p_radii,
                                    p_velocity = p_velocity,
                                    p_logging = p_logging,
                                    **p_kwargs )
        




## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class StreamMLProClouds3D8C10000Dynamic (StreamMLProClouds):
    """
    This benchmark stream generates 10000 3-dimensional instances that form 8 dynamic random point clouds.

    See also: class StreamMLProClouds

    Parameters
    ----------
    p_radii : list
        Radii of the clouds. Default = 100.
    p_seed 
        Seeding value for the random generator. Default = None (no seeding).
    p_logging
        Log level (see constants of class Log). Default: Log.C_LOG_ALL.
    """

    C_ID                    = 'Clouds3D8C10000Dynamic'
    C_NAME                  = 'Dynamic Clouds 3D'
    C_VERSION               = '1.0.1'
    C_NUM_DIMENSIONS        = 3
    C_NUM_INSTANCES         = 10000
    C_SCIREF_ABSTRACT       = 'Demo stream provides 10000 3D instances randomly positioned around eight randomly moving centers.'
    C_BOUNDARIES            = [-1000,1000]

## -------------------------------------------------------------------------------------------------
    def __init__( self,
                  p_radii : list = [100.0],
                  p_velocity : float = 1,
                  p_logging = Log.C_LOG_ALL,
                  **p_kwargs ):
        
        StreamMLProClouds.__init__( self,
                                    p_num_dim = self.C_NUM_DIMENSIONS,
                                    p_num_instances = self.C_NUM_INSTANCES,
                                    p_num_clouds = 8,
                                    p_radii = p_radii,
                                    p_velocity = p_velocity,
                                    p_logging = p_logging,
                                    **p_kwargs )
    
