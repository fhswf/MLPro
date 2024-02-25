## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - A Synoptic Framework for Standardized Machine Learning Tasks
## -- Package : mlpro
## -- Module  : test_math.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2021-05-23  1.0.0     DA       Creation
## -- 2021-09-11  1.0.0     MRD      Change Header information to match our new library name
## -------------------------------------------------------------------------------------------------

"""
Ver. 1.0.0 (2021-05-23)

Unit test classes for basic mathematical functions.
"""


import unittest
import numpy as np
from mlpro.bf.math import Dimension, ESpace, Element



## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class TestMath(unittest.TestCase):
    """
    Unit tests for module math.py.
    """

## -------------------------------------------------------------------------------------------------
    def test_ESpace(self):
 
        d1      = Dimension('x1', 'Variable x1', 'mm', [-5,5])
        d2      = Dimension('x2', 'Variable x2', 'mm', [-15,15])
        d3      = Dimension('x3', 'Variable x3', 'mm', [-15,25])

        espace  = ESpace()
        espace.add_dim(d1)
        espace.add_dim(d2)
        espace.add_dim(d3)
        self.assertEqual(espace.get_num_dim(),3, 'Dimension of example space is wrong')

        e1      = Element(espace)
        e2      = Element(espace)

        e1.set_values(np.array([1,2,3]))
        e2.set_values(np.array([4,5,6]))

        dist    = espace.distance(e1,e2)
        self.assertAlmostEqual(dist, 5.1961524, 7, 'Distance computation in Euclidian space failed')



if __name__ == '__main__': unittest.main()
