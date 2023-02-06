## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - A Synoptic Framework for Standardized Machine Learning Tasks
## -- Package : mlpro.bf.examples
## -- Module  : howto_bf_physics_002_unit_converter.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2023-01-15  0.0.0     SY       Creation
## -- 2023-01-15  1.0.0     SY       Release of first version
## -- 2023-01-16  1.0.1     SY       Renaming and debugging
## -- 2023-02-04  1.0.2     SY       Shift UnitConverter from bf.math to bf.physics
## -------------------------------------------------------------------------------------------------


"""
Ver. 1.0.2 (2023-02-04)

This module provides an example of using the unit converter in MLPro.

You will learn:

1) How to use the the unit converter

"""


from mlpro.bf.physics.unitconverter import UnitConverter
from mlpro.bf.various import Log



if __name__ == "__main__":
    p_print = True
else:
    p_print = False

# 1 Initialize unit converters
conv_length = UnitConverter(p_name='conv_length',
                            p_type=UnitConverter.C_UNIT_CONV_LENGTH,
                            p_unit_in='m',
                            p_unit_out='km')

conv_pressure = UnitConverter(p_name='conv_pressure',
                              p_type=UnitConverter.C_UNIT_CONV_PRESSURE,
                              p_unit_in='bar',
                              p_unit_out='Pa')

conv_current = UnitConverter(p_name='conv_current',
                             p_type=UnitConverter.C_UNIT_CONV_CURRENT,
                             p_unit_in='mA',
                             p_unit_out='A')

conv_force = UnitConverter(p_name='conv_force',
                           p_type=UnitConverter.C_UNIT_CONV_FORCE,
                           p_unit_in='N',
                           p_unit_out='J/cm')

conv_power = UnitConverter(p_name='conv_power',
                           p_type=UnitConverter.C_UNIT_CONV_POWER,
                           p_unit_in='W',
                           p_unit_out='kW')

conv_mass = UnitConverter(p_name='conv_mass',
                          p_type=UnitConverter.C_UNIT_CONV_MASS,
                          p_unit_in='kg',
                          p_unit_out='lb')

conv_time = UnitConverter(p_name='conv_time',
                          p_type=UnitConverter.C_UNIT_CONV_TIME,
                          p_unit_in='hr',
                          p_unit_out='s')

conv_temperature = UnitConverter(p_name='conv_temperature',
                                 p_type=UnitConverter.C_UNIT_CONV_TEMPERATURE,
                                 p_unit_in='K',
                                 p_unit_out='F')


# 2. Call the defined unit converters
conv_set = [conv_length,
            conv_pressure,
            conv_current,
            conv_force,
            conv_power,
            conv_mass,
            conv_time,
            conv_temperature]
p_input = 10

for conv in conv_set:
    output = conv(p_input)
    if p_print:
        print('We convert %.1f%s to %.2f%s'%(p_input, conv._unit_in, output, conv._unit_out))