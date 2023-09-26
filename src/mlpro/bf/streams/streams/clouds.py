## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - A Synoptic Framework for Standardized Machine Learning Tasks
## -- Package : mlpro.bf.streams.streams
## -- Module  : clouds.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2023-08-29  0.0.0     SR       Creation
## -- 2023-08-29  1.0.0     SR       First draft implementation
## -- 2023-09-15  1.0.1     LSB      Bug Fix
## -------------------------------------------------------------------------------------------------

"""
Ver. 1.0.1 (2023-09-15)

This module provides the native stream classes StreamMLProClouds, StreamMLProClouds2D4C1000Static,
StreamMLProClouds3D8C2000Static, StreamMLProClouds2D4C5000Dynamic and StreamMLProClouds3D8C10000Dynamic.
These stream provides instances with self.C_NUM_DIMENSIONS dimensional random feature data, placed around
centers (can be defined by user) which may or maynot move over time.

"""


import random
from mlpro.bf.streams.models import *
from mlpro.bf.streams.streams.provider_mlpro import StreamMLProBase



## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class StreamMLProClouds (StreamMLProBase):
    """
    This demo stream provides self.C_NUM_INSTANCES n-dimensional instances randomly positioned around
    centers which may or may not move over time.

    p_num_dim : int
        The number of dimensions or features of the data. Default = 3.
    p_num_instances : int
        Total number of instances. The value '0' means indefinite. Default = 1000.
    p_num_clouds : int
        Number of clouds. Default = 4.
    p_radii : list
        Radii of the clouds. Default = 100.
    p_behaviour : str
        Type of the clouds - static or dynamic. Default = 'dynamic'
    p_velocity : foat
        Velocity factor for the centers. Default = 0.1.
    p_logging
        Log level (see constants of class Log). Default: Log.C_LOG_ALL.
    """

    C_ID                    = 'CloudsNDim'
    C_NAME                  = 'Clouds N-Dim'
    C_TYPE                  = 'Demo'
    C_VERSION               = '1.0.0'
    C_BEHAVIOUR             = ['static', 'dynamic']
    C_SCIREF_ABSTRACT       = 'Demo stream provides self.C_NUM_INSTANCES C_NUM_DIMENSIONS-dimensional instances per cluster randomly positioned around centers which may or maynot move over time.'
    C_BOUNDARIES            = [-1000,1000]

## -------------------------------------------------------------------------------------------------
    def __init__( self,
                  p_num_dim : int = 3,
                  p_num_instances : int = 1000,
                  p_num_clouds : int = 8,
                  p_radii : list = [100.0],
                  p_behaviour : str = 'dynamic',
                  p_velocity : float = 0.1,
                  p_logging = Log.C_LOG_ALL,
                  **p_kwargs ):
        
        if str.lower(p_behaviour) not in self.C_BEHAVIOUR:
            raise ValueError(f"Invalid value for behaviour, allowed values are {self.C_BEHAVIOUR}")
           
        self.num_dim = int(p_num_dim)
        self.radii = p_radii
        self.num_clouds = int(p_num_clouds)
        self.velocity = p_velocity
        self.centers = []
        self.centers_step = []
        self.C_NUM_INSTANCES = p_num_instances
        self.behaviour = str.lower(p_behaviour)


        StreamMLProBase.__init__(self,
                                 p_logging=p_logging,
                                 **p_kwargs)
        

## -------------------------------------------------------------------------------------------------
    def _setup_feature_space(self) -> MSpace:
        feature_space : MSpace = MSpace()
        """
        """
        for i in range(self.num_dim):
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

        # Preparation
        try:
            seed = Stream.set_random_seed(p_seed=32)
        except:
            seed = random.seed(32)

        # Compute the initial positions of the centers
        for i in range(self.num_clouds):

            self.centers.append([])

            for j in range(self.num_dim):
                self.centers[i].append(random.randint(self.C_BOUNDARIES[0],
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
                    mag = mag + (self.centers[x][i]-self.centers_step[x][0])**2
                mag = mag**0.5
                if mag != 0:
                    for j in range(self.num_dim):
                        self.centers_step[x][j] = ((self.centers[x][j]-self.centers_step[x][j])/ mag)*self.velocity
                else:
                    for j in range(self.num_dim):
                        self.centers_step[x][j] = (1/(self.num_dim**0.5))*self.velocity

## -------------------------------------------------------------------------------------------------
    def _get_next(self) -> Instance:

        if self.C_NUM_INSTANCES== 0: pass

        elif self._index == self.C_NUM_INSTANCES: raise StopIteration

        if self.behaviour == 'dynamic':
            self.centers = self.centers + self.centers_step

        id = random.randint(0, self.num_clouds)
        if len(self.radii) == 1:
            radius = self.radii[0]
        else:
            radius = self.radii[id]

        instance = []

        for i in range(self.num_dim):
            instance.append((random.random())*radius)
            a = random.random()
            if a<0.5:
                instance[i] = instance[i]*(-1)

        instance = instance + self.centers[id]

        feature_data = Element(self._feature_space)
        feature_data.set_values(p_values=instance)

        self._index += 1

        return Instance( p_feature_data=feature_data )
    

## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class StreamMLProClouds2D4C1000Static (StreamMLProClouds):

    C_ID                    = 'StreamMLProClouds2D4C1000Static'
    C_NAME                  = 'Static Clouds 2D'
    C_TYPE                  = 'Demo'
    C_VERSION               = '1.0.0'
    C_NUM_DIMENSIONS        = 2
    C_NUM_INSTANCES         = 1000
    C_SCIREF_ABSTRACT       = 'Demo stream provides 1000 2D instances randomly positioned around four fixed centers.'
    C_BOUNDARIES            = [-1000,1000]

## -------------------------------------------------------------------------------------------------
    def __init__( self,
                  p_radii : list = [100.0],
                  p_velocity : float = 0.1,
                  p_logging = Log.C_LOG_ALL,
                  **p_kwargs ):
        
        StreamMLProClouds.__init__(self,
                                 p_logging=p_logging,
                                 **p_kwargs)
        
        self.num_dim = 2
        self.radii = p_radii
        self.num_clouds = 4
        self.velocity = p_velocity
        self.cloud_centers = []
        self.behaviour = 'static'
        self.centers_step = []
    


## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class StreamMLProClouds3D8C2000Static (StreamMLProClouds):

    C_ID                    = 'StreamMLProClouds3D8C2000Static'
    C_NAME                  = 'Static Clouds 3D'
    C_TYPE                  = 'Demo'
    C_VERSION               = '1.0.0'
    C_NUM_DIMENSIONS        = 3
    C_NUM_INSTANCES         = 2000
    C_SCIREF_ABSTRACT       = 'Demo stream provides 2000 3D instances randomly positioned around eight fixed centers.'
    C_BOUNDARIES            = [-1000,1000]

## -------------------------------------------------------------------------------------------------
    def __init__( self,
                  p_radii : list = [100.0],
                  p_velocity : float = 0.1,
                  p_logging = Log.C_LOG_ALL,
                  **p_kwargs ):
        
        StreamMLProClouds.__init__(self,
                                 p_logging=p_logging,
                                 **p_kwargs)
        
        self.num_dim = 3
        self.radii = p_radii
        self.num_clouds = 8
        self.velocity = p_velocity
        self.cloud_centers = []
        self.behaviour = 'static'
        self.centers_step = []
    


## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class StreamMLProClouds2D4C5000Dynamic (StreamMLProClouds):

    C_ID                    = 'StreamMLProClouds2D4C5000Dynamic'
    C_NAME                  = 'Dynamic Clouds 2D'
    C_TYPE                  = 'Demo'
    C_VERSION               = '1.0.0'
    C_NUM_DIMENSIONS        = 2
    C_NUM_INSTANCES         = 0
    C_SCIREF_ABSTRACT       = 'Demo stream provides 2000 2D instances randomly positioned around four randomly moving centers.'
    C_BOUNDARIES            = [-1000,1000]

## -------------------------------------------------------------------------------------------------
    def __init__( self,
                  p_radii : list = [100.0],
                  p_velocity : float = 0.1,
                  p_logging = Log.C_LOG_ALL,
                  **p_kwargs ):
        
        StreamMLProClouds.__init__(self,
                                 p_logging=p_logging,
                                 **p_kwargs)
        
        self.num_dim = 2
        self.radii = p_radii
        self.num_clouds = 4
        self.velocity = p_velocity
        self.cloud_centers = []
        self.behaviour = 'dynamic'
        self.centers_step = []
    


## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class StreamMLProClouds3D8C10000Dynamic (StreamMLProClouds):

    C_ID                    = 'StreamMLProClouds3D8C10000Dynamic'
    C_NAME                  = 'Static Clouds 2D'
    C_TYPE                  = 'Demo'
    C_VERSION               = '1.0.0'
    C_NUM_DIMENSIONS        = 3
    C_NUM_INSTANCES         = 10000
    C_SCIREF_ABSTRACT       = 'Demo stream provides 10000 3D instances randomly positioned around eight randomly moving centers.'
    C_BOUNDARIES            = [-1000,1000]

## -------------------------------------------------------------------------------------------------
    def __init__( self,
                  p_radii : list = [100.0],
                  p_velocity : float = 0.1,
                  p_logging = Log.C_LOG_ALL,
                  **p_kwargs ):
        
        StreamMLProClouds.__init__(self,
                                 p_logging=p_logging,
                                 **p_kwargs)
        
        self.num_dim = 3
        self.radii = p_radii
        self.num_clouds = 8
        self.velocity = p_velocity
        self.cloud_centers = []
        self.behaviour = 'dynamic'
        self.centers_step = []
    
