## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - A Synoptic Framework for Standardized Machine Learning Tasks
## -- Package : mlpro
## -- Module  : howto_bf_003_spaces_and_elements.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2021-05-28  1.0.0     DA       Creation/Release
## -- 2021-09-11  1.0.0     MRD      Change Header information to match our new library name
## -- 2021-09-23  1.0.1     DA       Adaption to changes in class Element
## -- 2021-12-03  1.0.2     DA       New method copy_append_spaces()
## -- 2022-02-25  1.0.3     SY       Refactoring due to auto generated ID in class Dimension
## -------------------------------------------------------------------------------------------------

"""
Ver. 1.0.3 (2022-02-25)

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
        self.copy_append_spaces()
        self.create_subspace1()
        self.create_subspace2()
        self.create_subspace3()
        self.create_element()
        self.change_elem_values()
        self.calculate_distance()


## -------------------------------------------------------------------------------------------------
    def create_euclidian_space(self):
        self.espace = ESpace()
        self.espace.add_dim(Dimension( p_name_short='Pos', p_name_long='Position', p_unit='m', p_unit_latex='m', p_boundaries=[0,100]))
        self.espace.add_dim(Dimension( p_name_short='Vel', p_name_long='Velocity', p_unit='m/s', p_unit_latex='\frac{m}{s}', p_boundaries=[-100,100]))
        self.espace.add_dim(Dimension( p_name_short='Acc', p_name_long='Acceleration', p_unit='m/qs', p_unit_latex='\frac{m}{s^2}', p_boundaries=[-100,100]))
        self.espace.add_dim(Dimension( p_name_short='Ang', p_name_long='Angle', p_unit='deg', p_unit_latex='deg', p_boundaries=[-45,45]))
        self.espace.add_dim(Dimension( p_name_short='AVel', p_name_long='Angle Velocity', p_unit='deg/s', p_unit_latex='\frac{deg}{s}', p_boundaries=[-100,100]))
        self.espace.add_dim(Dimension( p_name_short='AAcc', p_name_long='Angle Acceleration', p_unit='deg/qs', p_unit_latex='\frac{deg}{s^2}', p_boundaries=[-100,100]))
        
        _ids = self.espace.get_dim_ids()
        self.C_POS       = _ids[0]
        self.C_VEL       = _ids[1]
        self.C_ACC       = _ids[2]
        self.C_ANG       = _ids[3]
        self.C_AVEL      = _ids[4]
        self.C_AACC      = _ids[5]
        
        self.log(self.C_LOG_TYPE_I, '6-dimensional Euclidian space created')


## -------------------------------------------------------------------------------------------------
    def copy_append_spaces(self):
        new_space = self.espace.copy(True)
        new_space.append(self.espace, p_new_dim_ids=False)
        self.log(self.C_LOG_TYPE_I, str(new_space.get_num_dim()) + '-dimensional Euclidian space created')


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
        e2.set_value(self.C_AACC,1)
        self.log(self.C_LOG_TYPE_I, 'New element e1 =', e1.get_values())
        self.log(self.C_LOG_TYPE_I, 'New element e2 =', e2.get_values())
        self.log(self.C_LOG_TYPE_I, 'Euclidian distance between e1 and e2 =', self.espace.distance(e1,e2))



if __name__ == "__main__":
    demo = MathDemo()
