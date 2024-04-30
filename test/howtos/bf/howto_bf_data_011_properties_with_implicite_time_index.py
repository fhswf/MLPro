## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - The integrative middleware framework for standardized machine learning
## -- Package : mlpro.bf.data
## -- Module  : howto_bf_data_011_properties_with_implicite_time_index.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2024-04-28  1.0.0     DA       Creation
## -------------------------------------------------------------------------------------------------

"""
Ver. 1.0.0 (2024-04-28)

This module demonstrates and validates the reuse of class mlpro.bf.data.Properties in own classes.
Properties are basically pairs of names and values at a well-defined time point. Additionally,
derivatives according to time are numerically approximated. The maximum order of derivation can
be defined individually for each property.

In this example, implicite integer pseudo time stamps are used.

You will learn:

1) How to inherit from MLPro's class mlpro.bf.data.Properties

2) How to define various types of properties

3) How to set the values of properties with implicite auto-derivation

"""


from mlpro.bf.data import Properties



# 1 Reuse MLPro's class Properties in your own custom class...
class MyDemo (Properties):

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



# 2 Instantiate an object from your class and define some properties
myobj = MyDemo()

# 2.1 Numerical property 'Position' with auto-derivation of a maximum order 2
myobj.define_property('Position', 2)

# 2.2 Numerical property 'Temperature' with auto-derivation of a maximum order 1
myobj.define_property('Temperature', 1)

# 2.3 Textual property 'Color'
myobj.define_property('Color')



# 3 Timepoint 1: Let's set initial values

# 3.1 A 3-dimensional tuple for property 'Position'...
myobj.set_property('Position', [1,2,3])

# 3.2 A scalar value for property 'Temperature'
myobj.set_property('Temperature', 20)

# 3.3 A string value for property 'Color'
myobj.set_property('Color', 'red')

myobj.print_properties()



# 4 Now let's auto-derive

# 4.1 New values
myobj.set_property('Position', [2,4,3])
myobj.set_property('Temperature', 22.5)
myobj.set_property('Color', 'yellow')

myobj.print_properties()



# 5 Now let's auto-derive once again

# 5.1 New values
myobj.set_property('Position', [7,3,-3])
myobj.set_property('Temperature', 27)
myobj.set_property('Color', 'green')

myobj.print_properties()



# 6 Last but not least: the dimensionality of our properties
for prop in myobj.get_properties().keys():
    print('Dimensionality of "' + prop + '":', myobj.get_property(prop).dim)