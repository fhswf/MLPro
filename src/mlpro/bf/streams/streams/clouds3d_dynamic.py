## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - A Synoptic Framework for Standardized Machine Learning Tasks
## -- Package : mlpro.bf.streams.streams
## -- Module  : clouds3d_dynamic.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2023-03-22  0.0.0     SR       Creation 
## -- 2023-03-22  1.0.0     SR       First draft implementation
## -- 2023-03-29  1.0.1     SP       Update to speed up the code
## -- 2023-04-07  1.0.2     SP       Add new parameter p_variance, update the parameter p_pattern with constants
## -- 2023-05-08  1.0.3     SP       Add new parameter p_no_clouds
## -------------------------------------------------------------------------------------------------

"""
Ver. 1.0.1 (2023-03-22)

This module provides the native stream class StreamMLProStaticClouds3D. This stream provides 250 
instances per cluster with 3-dimensional random feature data placed around centers (can be defined by user) which move over time.
"""

import numpy as np
from mlpro.bf.streams.models import *
from mlpro.bf.streams.streams.provider_mlpro import StreamMLProBase




## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class StreamMLProDynamicClouds3D (StreamMLProBase):
    """
    This demo stream provides 250 3-dimensional instances per cluster randomly positioned around centers which move over time.
    """

    C_ID                = 'DynamicClouds3D'
    C_NAME              = 'Dynamic Clouds 3D'
    C_TYPE              = 'Demo'
    C_VERSION           = '1.0.0'
    C_NUM_INSTANCES     = '2000 by default'
    C_SCIREF_ABSTRACT   = 'Demo stream provides 250 3-dimensional instances per cluster randomly positioned around centers which move over time.'
    C_BOUNDARIES        = [-100,100]
    C_PATTERN           = ['random', 'random chain', 'static', 'merge']

## -------------------------------------------------------------------------------------------------
    def __init__(self, p_pattern='random', p_no_clouds=8, p_variance=5.0, p_logging=Log.C_LOG_ALL, **p_kwargs):
        StreamMLProBase.__init__(self, pattern='random', p_logging=Log.C_LOG_ALL, **p_kwargs)
        if str.lower(p_pattern) not in self.C_PATTERN:
            raise ValueError(f"Invalid value for pattern, allowed values are {self.ALLOWED_VALUES}")
        self.pattern = str.lower(p_pattern)
        self.variance = p_variance
        self.no_clouds = int(p_no_clouds)
        self.num_instances = 250*self.no_clouds


## -------------------------------------------------------------------------------------------------
    def _setup_feature_space(self) -> MSpace:
        feature_space : MSpace = MSpace()

        for i in range(3):
            feature_space.add_dim( Feature( p_name_short = 'f' + str(i),
                                            p_base_set = Feature.C_BASE_SET_R,
                                            p_name_long = 'Feature #' + str(i),
                                            p_name_latex = '',
                                            p_boundaries = self.C_BOUNDARIES,
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

        self._dataset = np.empty( (self.num_instances, 3))


        # Compute the initial positions of the centers
        centers = np.random.RandomState(seed=seed).randint(self.C_BOUNDARIES[0], self.C_BOUNDARIES[1], size=(self.no_clouds, 3))

        if self.pattern == 'random':
            # Compute the final positions of the centers
            final_centers = np.random.RandomState(seed=seed).randint(self.C_BOUNDARIES[0], self.C_BOUNDARIES[1], size=(self.no_clouds, 3))

        elif self.pattern == 'random chain':
            # Compute the final positions of the centers
            final_centers = np.zeros((self.no_clouds, 3))
            final_centers[0] = centers[-1]
            final_centers[1:] = centers[:-1]

        elif self.pattern == 'static':
            # Use the initial positions as the final positions
            final_centers = centers

        elif self.pattern == 'merge':
            # Compute the final positions of the centers
            final_centers = np.zeros((self.no_clouds, 3))
            if self.no_clouds%2==0:
                e1 = self.no_clouds
                e2 = 0
                m = int(e1/2)
            else:
                e1 = self.no_clouds-1
                e2 = self.no_clouds
                m = int(e1/2)
                
            final_centers[:m] = np.random.RandomState(seed=seed).randint(self.C_BOUNDARIES[0], self.C_BOUNDARIES[1], size=(m, 3))
            final_centers[m:e1] = final_centers[:m]
            if e2!=0:
                final_centers[e2] = final_centers[m]


        # 2 Create 250 noisy inputs around each of the 8 hotspots
        a = np.random.RandomState(seed=seed).rand(self.num_instances, 3)**3
        s = np.round(np.random.RandomState(seed=seed).rand(self.num_instances, 3))
        s[s==0] = -1
        fx = self.variance
        c = a*s * np.array([fx, fx, fx]) 
        
        # Create the dataset
        dataset = np.zeros((self.num_instances, 3))

        if self.pattern == 'static':
            centers_diff = (0-centers) / 125

            i = 0
            while i<125:
                dataset[i*self.no_clouds:(i+1)*self.no_clouds] = centers + c[i*self.no_clouds:(i+1)*self.no_clouds]
                centers = centers + centers_diff
                i += 1

            while i<250:
                dataset[i*self.no_clouds:(i+1)*self.no_clouds] = centers + c[i*self.no_clouds:(i+1)*self.no_clouds]
                centers = centers - centers_diff
                i += 1

        else:
            centers_diff = (final_centers - centers) / 250

            for i in range(250):
                dataset[i*self.no_clouds:(i+1)*self.no_clouds] = centers + c[i*self.no_clouds:(i+1)*self.no_clouds]
                centers = centers + centers_diff

        self._dataset = dataset

## -------------------------------------------------------------------------------------------------
    def set_random_seed(self, p_seed=None):
        self._random_seed = p_seed

