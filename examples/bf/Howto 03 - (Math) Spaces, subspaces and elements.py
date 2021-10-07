## -------------------------------------------------------------------------------------------------
## -- Project : FH-SWF Automation Technology - Common Code Base (CCB)
## -- Package : mlpro
## -- Module  : Howto 03 - (Math) Spaces, subspaces and elements
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2021-05-28  1.0.0     DA       Creation/Release
## -- 2021-09-11  1.0.0     MRD      Change Header information to match our new library name
## -- 2021-09-23  1.0.1     DA       Adaption to changes in class Element
## -------------------------------------------------------------------------------------------------

"""
Ver. 1.0.1 (2021-09-23)

This module demonstrates how to create a space and subspaces and to spawn elements.
"""


from mlpro.bf.various import Log
from mlpro.bf.math import *



## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class MathDemo(Log):

    C_TYPE      = 'Demo'
    C_NAME      = 'Spaces & Elements'

    # Some constants for dimension indices to make it more understandable 
    C_POS       = 0
    C_VEL       = 1
    C_ACC       = 2
    C_ANG       = 3
    C_AVEL      = 4
    C_AACC      = 5

## -------------------------------------------------------------------------------------------------
    def __init__(self, p_logging=True):
        super().__init__(p_logging=p_logging)
        self.create_euclidian_space()
        self.create_subspace1()
        self.create_subspace2()
        self.create_subspace3()
        self.create_element()
        self.change_elem_values()
        self.calculate_distance()


## -------------------------------------------------------------------------------------------------
    def create_euclidian_space(self):
        self.espace = ESpace()
        self.espace.add_dim(Dimension(self.C_POS, 'Pos', 'Position', '', 'm', 'm', [0,100]))
        self.espace.add_dim(Dimension(self.C_VEL, 'Vel', 'Velocity', '', 'm/s', '\frac{m}{s}', [-100,100]))
        self.espace.add_dim(Dimension(self.C_ACC, 'Acc', 'Acceleration', '', 'm/qs', '\frac{m}{s^2}', [-100,100]))
        self.espace.add_dim(Dimension(self.C_ANG, 'Ang', 'Angle', '', 'deg', 'deg', [-45,45]))
        self.espace.add_dim(Dimension(self.C_AVEL, 'AVel', 'Angle Velocity', '', 'deg/s', '\frac{deg}{s}', [-100,100]))
        self.espace.add_dim(Dimension(self.C_AACC, 'AAcc', 'Angle Acceleration', '', 'deg/qs', '\frac{deg}{s^2}', [-100,100]))
        self.log(self.C_LOG_TYPE_I, '6-dimensional Euclidian space created')


## -------------------------------------------------------------------------------------------------
    def create_subspace1(self):
        self.subspace1 = self.espace.spawn([self.C_POS, self.C_VEL, self.C_ACC])
        self.log(self.C_LOG_TYPE_I, 'Subspace 1 - Number of dimensions and short name of second dimension:', self.subspace1.get_num_dim(), self.subspace1.get_dim(self.C_VEL).get_name_short())


## -------------------------------------------------------------------------------------------------
    def create_subspace2(self):
        self.subspace2 = self.espace.spawn([self.C_ANG, self.C_AVEL, self.C_AACC])
        self.log(self.C_LOG_TYPE_I, 'Subspace 2 - Number of dimensions and short name of third dimension:', self.subspace2.get_num_dim(), self.subspace2.get_dim(self.C_AACC).get_name_short())


## -------------------------------------------------------------------------------------------------
    def create_subspace3(self):
        self.subspace3 = self.espace.spawn([self.C_POS, self.C_ANG])
        self.log(self.C_LOG_TYPE_I, 'Subspace 3 - Number of dimensions and short name of second dimension:', self.subspace3.get_num_dim(), self.subspace3.get_dim(self.C_ANG).get_name_short())


## -------------------------------------------------------------------------------------------------
    def create_element(self):
        self.elem = Element(self.espace)
        self.log(self.C_LOG_TYPE_I, 'New element created with dim ids / values:', self.elem.get_dim_ids(), ' / ', self.elem.get_values())


## -------------------------------------------------------------------------------------------------
    def change_elem_values(self):
        # Changing a value indexed by a unique dimension index...
        self.elem.set_value(self.C_POS, 4.77)
        self.elem.set_value(self.C_VEL, -8.22)
        self.log(self.C_LOG_TYPE_I, 'Element changed to ', self.elem.get_values())


## -------------------------------------------------------------------------------------------------
    def calculate_distance(self):
        e1 = Element(self.espace)
        e2 = Element(self.espace)
        e2.set_value(5,1)
        self.log(self.C_LOG_TYPE_I, 'New element e1 =', e1.get_values())
        self.log(self.C_LOG_TYPE_I, 'New element e2 =', e2.get_values())
        self.log(self.C_LOG_TYPE_I, 'Euclidian distance between e1 and e2 =', self.espace.distance(e1,e2))



demo = MathDemo()
