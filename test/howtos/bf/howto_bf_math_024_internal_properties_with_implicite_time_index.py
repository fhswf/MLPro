## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - The integrative middleware framework for standardized machine learning
## -- Package : mlpro.bf.math
## -- Module  : howto_bf_math_024_internal_properties_with_implicite_time_index.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2024-05-05  1.0.0     DA       Creation
## -------------------------------------------------------------------------------------------------

"""
Ver. 1.0.0 (2024-05-05)

This module demonstrates and validates the reuse of class mlpro.bf.data.Properties in own classes.
Properties are basically pairs of names and values at a well-defined time point. Additionally,
derivatives according to time are numerically approximated. The maximum order of derivation can
be defined individually for each property.

In this example, properties are added by the custom class during instantiation. Furthermore, implicite 
integer pseudo time stamps are used.

You will learn:

1) How to inherit from MLPro's class mlpro.bf.data.Properties

2) How to define various types of properties

3) How to set the values of properties with implicite auto-derivation

"""


from mlpro.bf.math.properties import *



# 1 Reuse MLPro's class Properties in your own custom class...
class MyDemo (Properties):

    C_PROPERTIY_DEFINITIONS : PropertyDefinitions = [ ['position', 2, Property],
                                                      ['temperature', 1, Property],
                                                      ['color', 0, Property] ]

    def __init__(self):
        super().__init__()
        self.add_properties(p_property_definitions=self.C_PROPERTIY_DEFINITIONS)

    def print_properties(self):
        properties = self.get_properties()

        for prop in properties.keys():
            print('Property "' + prop + '" at time stamp ', properties[prop].time_stamp)
            print('   Value:', properties[prop].value)
            print('   Derivatives:')
            order = 0
            while True:
                try:
                    print('       Order', order, ': ', properties[prop].derivatives[order])
                except:
                    break

                order += 1

            print()



# 2 Instantiate an object from your class with implicite property definition
myobj = MyDemo()



# 3 Timepoint 1: Let's set initial values

# 3.1 A 3-dimensional tuple for property 'position'...
myobj.position.set([1,2,3])

# 3.2 A scalar value for property 'temperature'
myobj.temperature.set(20)

# 3.3 A string value for property 'color'
myobj.color.set('red')

myobj.print_properties()



# 4 Now let's auto-derive

# 4.1 New values
myobj.position.set([2,4,3])
myobj.temperature.set(22.5)
myobj.color.set('yellow')

myobj.print_properties()



# 5 Now let's auto-derive once again

# 5.1 New values
myobj.position.set([7,3,-3])
myobj.temperature.set(27)
myobj.color.set('green')


myobj.print_properties()



# 6 Last but not least: the dimensionality of our properties
for prop_name, prop in myobj.get_properties().items():
    print('Dimensionality of "' + prop_name + '":', prop.dim)
