## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - A Synoptic Framework for Standardized Machine Learning Tasks
## -- Package : mlpro.bf.ui.scui.pool
## -- Module  : iisbenchmark
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2021-06-27  0.0.0     DA       Creation
## -- 2021-mm-dd  1.0.0     DA       Released first version
## -- 2021-09-11  1.0.0     MRD      Change Header information to match our new library name
## -- 2022-10-08  1.0.1     DA       Refactoring following class Dimensions update 
## -------------------------------------------------------------------------------------------------

"""
Ver. 1.0.1 (2022-10-08)

This module provides benchmark test implementations for use within the Interactive Input Space (IIS).
"""


#from typing import ValuesView
import numpy as np
from numpy.random import default_rng
from math import sin, cos, pi
from mlpro.bf.various import Log
import random





## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class IISBenchmark (Log): 
    """
    Root class for a IIS benchmark test.
    """

    C_TYPE          = 'IIS Benchmark'
    C_GROUP         = 'IIS Buildin'
    C_NAME          = 'Rename me!'
    C_HORIZON       = 100
    C_INPUTS        = 100
    C_DESCRIPTION   = 'This is a demo description.\nDescription line 2\nDescription line3.'
    
## -------------------------------------------------------------------------------------------------
    def __init__(self, p_ispace, p_iis_input_cb, p_logging=True) -> None:
        super().__init__(p_logging)
        self.ispace         = p_ispace
        self.iis_input_cb   = p_iis_input_cb
        self.num_dim        = p_ispace.get_num_dim()
        self.input          = np.zeros(self.num_dim)
        self.reset()


## -------------------------------------------------------------------------------------------------
    def reset(self): pass


## -------------------------------------------------------------------------------------------------
    def run(self): 
        self.log(self.C_LOG_TYPE_I, 'Start')






## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class IISBenchmark2D: 
    """
    Marker class for 2D benchmark tests.
    """
    pass




## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class IISBenchmark3D: 
    """
    Marker class for 3D benchmark tests.
    """
    pass




## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class IISBMRandom(IISBenchmark):
    """
    ...
    """
    
    C_GROUP         = 'IIS Buildin'
    C_NAME          = 'Random Inputs'
    C_DESCRIPTION   = 'This benchmark test generates a reproducible set of random inputs.'
 
## -------------------------------------------------------------------------------------------------
    def set_num_inputs(self, p_num):
        self.num_inputs = p_num


## -------------------------------------------------------------------------------------------------
    def reset(self):
        super().reset()

        self.x_min      = np.zeros(self.num_dim)
        self.x_factor   = np.zeros(self.num_dim)

        for i, dim_id in enumerate(self.ispace.get_dim_ids()):
            boundaries          = self.ispace.get_dim(dim_id).get_boundaries()
            self.x_min[i]       = boundaries[0]
            self.x_factor[i]    = (boundaries[1] - boundaries[0])

        self.inputs     = np.random.RandomState(seed=0).rand(self.num_inputs, self.num_dim) * self.x_factor + self.x_min


## -------------------------------------------------------------------------------------------------
    def run(self):
        super().run()

        for i in range(self.num_inputs):
            self.iis_input_cb(self.inputs[i])





## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class IISBMRandom10(IISBMRandom, IISBenchmark2D, IISBenchmark3D): 

    C_NAME          = '10 Random Inputs'
    C_HORIZON       = 100
    C_INPUTS        = 10

## -------------------------------------------------------------------------------------------------
    def reset(self):
        self.set_num_inputs(self.C_INPUTS)
        IISBMRandom.reset(self)





## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class IISBMRandom100(IISBMRandom, IISBenchmark2D, IISBenchmark3D):

    C_NAME          = '100 Random Inputs'
    C_HORIZON       = 100
    C_INPUTS        = 100

## -------------------------------------------------------------------------------------------------
    def reset(self):
        self.set_num_inputs(self.C_INPUTS)
        IISBMRandom.reset(self)





## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class IISBMRandom1000(IISBMRandom, IISBenchmark2D, IISBenchmark3D):

    C_NAME          = '1000 Random Inputs'
    C_HORIZON       = 1000
    C_INPUTS        = 1000

## -------------------------------------------------------------------------------------------------
    def reset(self):
        self.set_num_inputs(self.C_INPUTS)
        IISBMRandom.reset(self)





## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class IISBMSpiral721(IISBenchmark, IISBenchmark2D):

    C_NAME          = 'Double Spiral (721 inputs)'
    C_DESCRIPTION   = 'This benchmark test generates 721 inputs positioned in a double spiral.'
    C_HORIZON       = 721
    C_INPUTS        = 721

## -------------------------------------------------------------------------------------------------
    def run(self):
        super().run()

        dims            = self.ispace.get_dims()

        x1_boundaries   = dims[0].get_boundaries()
        x2_boundaries   = dims[1].get_boundaries()

        center_x1       = ( (x1_boundaries[1] - x1_boundaries[0]) / 2 ) + x1_boundaries[0]
        center_x2       = ( (x2_boundaries[1] - x2_boundaries[0]) / 2 ) + x2_boundaries[0]
        
        radius_x1       = (x1_boundaries[1] - x1_boundaries[0]) / 2
        radius_step_x1  = radius_x1 / 360
        radius_x2       = (x2_boundaries[1] - x2_boundaries[0]) / 2
        radius_step_x2  = radius_x2 / 360
        
        radius_sign = 1
        
        for i in range(self.C_INPUTS):
            
            bm = i *2 * pi / 180
            x1 = cos(bm) * radius_x1 * radius_sign + center_x1
            x2 = sin(bm) * radius_x2 + center_x2
            
            self.iis_input_cb([x1, x2])
            
            radius_x1 -= radius_step_x1
            radius_x2 -= radius_step_x2
            if radius_x1 < 0:
                radius_x1       = 0
                radius_step_x1  *= -1
                radius_x2       = 0
                radius_step_x2  *= -1
                radius_sign     = -1        





## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class IISBM4Hotspots(IISBenchmark, IISBenchmark2D):

    C_NAME          = '1000 Random inputs around 4 hotspots'
    C_DESCRIPTION   = 'This test positions 4 hotspots and generates 1000 random inputs around them.'
    C_INPUTS        = 1000

## -------------------------------------------------------------------------------------------------
    def run(self):
        # 1 Create 4 fixed hotspots
        hotspots      = []

        dims            = self.ispace.get_dims()

        x1_boundaries   = dims[0].get_boundaries()
        x2_boundaries   = dims[1].get_boundaries()
        
        dx1             = ( x1_boundaries[1] - x1_boundaries[0] ) / 4
        x1_1            = x1_boundaries[0] + dx1
        x1_2            = x1_boundaries[1] - dx1
        
        dx2             = ( x2_boundaries[1] - x2_boundaries[0] ) / 4
        x2_1            = x2_boundaries[0] + dx2
        x2_2            = x2_boundaries[1] - dx2
        
        hotspots        = [ [ x1_1, x2_1 ], [ x1_2, x2_1 ], [ x1_1, x2_2 ], [ x1_2, x2_2 ] ]
       
        
        # 2 Create 250 noisy inputs around each of the 4 fixed hotspots
        random.seed()
        
        radius_max_x1   = dx1
        radius_max_x2   = dx2
        bm_aux          = 2 * np.pi
        
        for i in range(int(self.C_INPUTS / 4)):
            for hsp in hotspots:
                bm          = random.random() * bm_aux
                radius_x1   = ( random.random() ** 2 ) * radius_max_x1
                radius_x2   = ( random.random() ** 2 ) * radius_max_x2
                
                x1  = hsp[0] + np.cos(bm) * radius_x1
                x2  = hsp[1] + np.sin(bm) * radius_x2 
                self.iis_input_cb([x1, x2])





## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class IISBM4Hotspots2(IISBenchmark, IISBenchmark2D):

    C_NAME          = '1000 Random inputs around 4 hotspots(2)'
    C_DESCRIPTION   = 'This test positions 4 hotspots and generates 1000 random inputs around them.'
    C_INPUTS        = 1000

## -------------------------------------------------------------------------------------------------
    def run(self):
        # 1 Create 4 fixed hotspots
        hotspots      = []

        dims            = self.ispace.get_dims()

        x1_boundaries   = dims[0].get_boundaries()
        x2_boundaries   = dims[1].get_boundaries()
        
        dx1             = ( x1_boundaries[1] - x1_boundaries[0] ) / 4
        x1_1            = x1_boundaries[0] + dx1
        x1_2            = x1_boundaries[1] - dx1
        
        dx2             = ( x2_boundaries[1] - x2_boundaries[0] ) / 4
        x2_1            = x2_boundaries[0] + dx2
        x2_2            = x2_boundaries[1] - dx2
        
        hotspots        = [ [ x1_1, x2_1 ], [ x1_2, x2_1 ], [ x1_1, x2_2 ], [ x1_2, x2_2 ] ]

        a = np.random.RandomState(seed=2).rand(self.C_INPUTS, 2)**3
        s = np.round(np.random.RandomState(seed=3).rand(self.C_INPUTS, 2))
        s[s==0] = -1
        fx1 = dx1 * 0.75
        fx2 = dx2 * 0.75
        c = a*s * np.array([fx1, fx2]) 
        
        j  = 0
        
        for i in range(int(self.C_INPUTS / 4)):
            for hsp in hotspots:
                x1  = hsp[0] + c[j][0]
                x2  = hsp[1] + c[j][1]
                self.iis_input_cb([x1, x2])
                j += 1




## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class IISBM4Hotspots3D(IISBenchmark, IISBenchmark3D):

    C_NAME          = '2000 Random inputs around 8 hotspots'
    C_DESCRIPTION   = 'This test positions 8 hotspots and generates 2000 random inputs around them.'
    C_INPUTS        = 2000

## -------------------------------------------------------------------------------------------------
    def run(self):
        # 1 Create 4 fixed hotspots
        hotspots      = []

        dims            = self.ispace.get_dims()

        x1_boundaries   = dims[0].get_boundaries()
        x2_boundaries   = dims[1].get_boundaries()
        x3_boundaries   = dims[2].get_boundaries()
        
        dx1             = ( x1_boundaries[1] - x1_boundaries[0] ) / 4
        x1_1            = x1_boundaries[0] + dx1
        x1_2            = x1_boundaries[1] - dx1
        
        dx2             = ( x2_boundaries[1] - x2_boundaries[0] ) / 4
        x2_1            = x2_boundaries[0] + dx2
        x2_2            = x2_boundaries[1] - dx2
 
        dx3             = ( x3_boundaries[1] - x3_boundaries[0] ) / 4
        x3_1            = x3_boundaries[0] + dx3
        x3_2            = x3_boundaries[1] - dx3
        
        hotspots        = [ [ x1_1, x2_1, x3_1 ], [ x1_2, x2_1, x3_1 ], [ x1_1, x2_2, x3_1 ], [ x1_2, x2_2, x3_1 ], [ x1_1, x2_1, x3_2 ], [ x1_2, x2_1, x3_2 ], [ x1_1, x2_2, x3_2 ], [ x1_2, x2_2, x3_2 ] ]

        a = np.random.RandomState(seed=2).rand(self.C_INPUTS, 3)**3
        s = np.round(np.random.RandomState(seed=3).rand(self.C_INPUTS, 3))
        s[s==0] = -1
        fx1 = dx1 * 0.75
        fx2 = dx2 * 0.75
        fx3 = dx3 * 0.75
        c = a*s * np.array([fx1, fx2, fx3]) 
        
        j  = 0
        
        for i in range(int(self.C_INPUTS / 8)):
            for hsp in hotspots:
                x1  = hsp[0] + c[j][0]
                x2  = hsp[1] + c[j][1]
                x3  = hsp[2] + c[j][2]
                self.iis_input_cb([x1, x2, x3])
                j += 1

