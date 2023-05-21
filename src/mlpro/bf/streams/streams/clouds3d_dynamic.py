## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - A Synoptic Framework for Standardized Machine Learning Tasks
## -- Package : mlpro.bf.streams.streams
## -- Module  : clouds3d_dynamic.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2023-03-22  0.0.0     SR       Creation 
## -- 2023-03-22  1.0.0     SR       First draft implementation
## -- 2023-03-29  1.0.1     SP       Updated to speed up the code
## -- 2023-04-07  1.0.2     SP       Added new parameter p_variance, update the parameter p_pattern with constants
## -- 2023-05-08  1.0.3     SP       Added new parameter p_no_clouds
## -- 2023-05-18  1.0.4     SP       Added new parameter p_velocity
## -- 2023-05-21  1.0.5     DA       - Completed the class documentation
## --                                - New attribute C_NUM_INST_PER_CLOUD
## --                                - Removed boundaries from feature definitions
## -------------------------------------------------------------------------------------------------

"""
Ver. 1.0.5 (2023-05-21)

This module provides the native stream class StreamMLProStaticClouds3D. This stream provides self.C_NUM_INSTANCES 
instances per cluster with 3-dimensional random feature data placed around centers (can be defined by user) which move over time.
"""

import numpy as np
from mlpro.bf.streams.models import *
from mlpro.bf.streams.streams.provider_mlpro import StreamMLProBase




## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class StreamMLProDynamicClouds3D (StreamMLProBase):
    """
    This demo stream provides self.C_NUM_INSTANCES 3-dimensional instances per cluster randomly positioned around centers which move over time.

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
    C_NUM_INSTANCES         = 0
    C_NUM_INST_PER_CLOUD    = 250
    C_SCIREF_ABSTRACT       = 'Demo stream provides self.C_NUM_INSTANCES 3-dimensional instances per cluster randomly positioned around centers which move over time.'
    C_BOUNDARIES            = [-60,60]
    C_PATTERN               = ['random', 'random chain', 'static', 'merge']

## -------------------------------------------------------------------------------------------------
    def __init__( self, 
                  p_pattern : str = 'random',
                  p_no_clouds : int = 8,
                  p_variance : float = 5.0,
                  p_velocity : float = 0.1,
                  p_logging = Log.C_LOG_ALL,
                  **p_kwargs ):
        
        StreamMLProBase.__init__(self,
                                 p_logging=p_logging,
                                 **p_kwargs)
        
        if str.lower(p_pattern) not in self.C_PATTERN:
            raise ValueError(f"Invalid value for pattern, allowed values are {self.C_PATTERN}")
        self.pattern = str.lower(p_pattern)
        self.variance = p_variance
        self.no_clouds = int(p_no_clouds)
        self.C_NUM_INSTANCES = self.C_NUM_INST_PER_CLOUD*self.no_clouds
        self.velocity = p_velocity


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

        self._dataset = np.empty( (self.C_NUM_INSTANCES, 3))


        # Compute the initial positions of the centers
        centers = np.random.RandomState(seed=seed).randint(self.C_BOUNDARIES[0],
                                                           self.C_BOUNDARIES[1], size=(self.no_clouds, 3))
        centers = centers.astype(np.float64)

        # Compute the final positions of the centers
        final_centers = np.random.RandomState(seed=seed).randint(self.C_BOUNDARIES[0],
                                                                    self.C_BOUNDARIES[1], size=(self.no_clouds, 3))
        final_centers = final_centers.astype(np.float64)

        for x in range(self.no_clouds):
            mag = ((centers[x][0]-final_centers[x][0])**2 + (centers[x][1]-final_centers[x][1])**2 + (centers[x][2]-final_centers[x][2])**2)**0.5
            if mag != 0:
                final_centers[x][:] = centers[x][:] + ((final_centers[x][:]-centers[x][:])/ mag)*self.C_NUM_INST_PER_CLOUD*self.velocity
            else:
                final_centers[x][:] = centers[x][:] + (1/(3**0.5))*self.C_NUM_INST_PER_CLOUD*self.velocity
            if x<(self.no_clouds-1) and self.pattern=='random chain':
                centers[x+1] = final_centers[x]

        if self.pattern == 'merge':
            if self.no_clouds%2==0:
                e1 = self.no_clouds
                e2 = 0
                m = int(e1/2)
            else:
                e1 = self.no_clouds-1
                e2 = e1
                m = int(e1/2)
            final_centers[m:e1] = final_centers[:m]
            if e2!=0:
                final_centers[e2] = final_centers[e1-1]

            for x in range(self.no_clouds-m):
                mag = ((centers[m+x][0]-final_centers[m+x][0])**2 + (centers[m+x][1]-final_centers[m+x][1])**2 + (centers[m+x][2]-final_centers[m+x][2])**2)**0.5
                if mag != 0:
                    centers[m+x][:] = final_centers[m+x][:] + ((centers[m+x][:] - final_centers[m+x][:])/ mag)*self.C_NUM_INST_PER_CLOUD*self.velocity
                else:
                    centers[m+x][:] = final_centers[m+x][:] + (1/(3**.5))*self.C_NUM_INST_PER_CLOUD*self.velocity


        # 2 Create self.C_NUM_INSTANCES noisy inputs around each of the 8 hotspots
        a = np.random.RandomState(seed=seed).rand(self.C_NUM_INSTANCES, 3)**3
        s = np.round(np.random.RandomState(seed=seed).rand(self.C_NUM_INSTANCES, 3))
        s[s==0] = -1
        fx = self.variance
        c = a*s * np.array([fx, fx, fx]) 
        
        # Create the dataset
        dataset = np.zeros((self.C_NUM_INSTANCES, 3))

        centers_diff = (final_centers - centers) / self.C_NUM_INST_PER_CLOUD


        if self.pattern == 'static':

            i = 0
            while i< (self.C_NUM_INST_PER_CLOUD / 2):
                dataset[i*self.no_clouds:(i+1)*self.no_clouds] = centers + c[i*self.no_clouds:(i+1)*self.no_clouds]
                centers = centers + centers_diff
                i += 1

            while i<self.C_NUM_INST_PER_CLOUD:
                dataset[i*self.no_clouds:(i+1)*self.no_clouds] = centers + c[i*self.no_clouds:(i+1)*self.no_clouds]
                centers = centers - centers_diff
                i += 1

        else:

            for i in range(self.C_NUM_INST_PER_CLOUD):
                dataset[i*self.no_clouds:(i+1)*self.no_clouds] = centers + c[i*self.no_clouds:(i+1)*self.no_clouds]
                centers = centers + centers_diff

        self._dataset = dataset


