## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - The integrative middleware framework for standardized machine learning
## -- Package : mlpro.bf.math
## -- Module  : howto_bf_math_021_external_properties_with_implicite_time_index.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2024-04-28  1.0.0     DA       Creation
## -- 2024-05-05  1.1.0     DA       Refactoring
## -- 2024-06-08  1.2.0     DA       Refactoring
## -------------------------------------------------------------------------------------------------

"""
Ver. 1.2.0 (2024-06-08)

This module demonstrates and validates the reuse of class mlpro.bf.data.Properties in own classes.
Properties are basically pairs of names and values at a well-defined time point. Additionally,
derivatives according to time are numerically approximated. The maximum order of derivation can
be defined individually for each property.

In this example, properties are added from outside to a custom class. Furthermore, implicite integer 
pseudo time stamps are used.

You will learn:

1) How to inherit from MLPro's class mlpro.bf.data.Properties

2) How to define various types of properties

3) How to set the values of properties with implicite auto-derivation

"""


from mlpro.bf.math.properties import Properties



# 1 Reuse MLPro's class Properties in your own custom class...
class MyDemo (Properties):

    def print_properties(self):
        properties = self.get_properties()

        for prop_name, (prop, link) in properties.items():
            print('Property "' + prop_name + '" at time stamp ', prop.time_stamp)
            print('   Value:', prop.value)
            print('   Derivatives:')
            order = 0
            while True:
                try:
                    print('       Order', order, ': ', prop.derivatives[order])
                except:
                    break

                order += 1

            print()



# 2 Instantiate an object from your class and define some properties
myobj = MyDemo()

# 2.1 Numerical property 'position' with auto-derivation of a maximum order 2
myobj.add_property('position', 2)

# 2.2 Numerical property 'temperature' with auto-derivation of a maximum order 1
myobj.add_property('temperature', 1)

# 2.3 Textual property 'color'
myobj.add_property('color')



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
for prop_name, (prop, link) in myobj.get_properties().items():
    print('Dimensionality of "' + prop_name + '":', prop.dim)
