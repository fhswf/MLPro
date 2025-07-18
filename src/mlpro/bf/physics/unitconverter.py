## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - The integrative middleware framework for standardized machine learning
## -- Package : mlpro.bf.physics
## -- Module  : unitconverter.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2023-01-16  0.0.0     SY       Creation
## -- 2023-01-16  1.0.0     SY       Shift UnitConverter from bf.systems
## -- 2023-01-18  1.0.1     SY       Debugging
## -- 2023-01-24  1.0.2     SY       Quality Assurance on TransferFunction
## -- 2023-02-04  1.0.3     SY       Shift UnitConverter from bf.math
## -- 2023-03-07  1.0.4     SY       Shift UnitConverter from bf.physics
## -------------------------------------------------------------------------------------------------


"""
Ver. 1.0.4 (2023-03-07)

This module provides models for unit conversions.
"""

from mlpro.bf.physics import TransferFunction
from mlpro.bf.various import Log



# Export list for public API
__all__ = [ 'UnitConverter' ]




## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------

class UnitConverter(TransferFunction):
    """
    This class serves as a base class of unit converters, which inherits the main attributes from
    a transfer function. By default, there are several ready-to-use unit converters available, such
    as, Length, Temperature, Pressure, Electric Current, Force, Power, Mass, and Time.
    
    Parameters
    ----------
    p_name : str
        name of the unit converter.
    p_id : int
        unique id of the unit converter. Default: None.
    p_type : int
        type of the unit converter. Default: None.
    p_unit_in : str
        unit of the unit converter's input. Default: None.
    p_unit_out : str
        unit of the unit converter's output. Default: None.
    p_logging
        Log level (see constants of class Log). Default: Log.C_LOG_ALL.
        
    Attributes
    ----------
    C_TYPE : str
        type of the base class. Default: 'UnitConverter'.
    C_NAME : str
        name of the unit converter. Default: ''.
    C_UNIT_CONV_LENGTH : int
        unit converter for length. Default: 0.
    C_UNIT_CONV_PRESSURE : int
        unit converter for pressure. Default: 1.
    C_UNIT_CONV_CURRENT : int
        unit converter for electric current. Default: 2.
    C_UNIT_CONV_FORCE : int
        unit converter for force. Default: 3.
    C_UNIT_CONV_POWER : int
        unit converter for power. Default: 4.
    C_UNIT_CONV_MASS : int
        unit converter for mass. Default: 5.
    C_UNIT_CONV_TIME : int
        unit converter for time. Default: 6.
    C_UNIT_CONV_TEMPERATURE : int
        unit converter for temperature. Default: 7.
    """

    C_TYPE                  = 'UnitConverter'
    C_NAME                  = ''
    C_UNIT_CONV_LENGTH      = 0
    C_UNIT_CONV_PRESSURE    = 1
    C_UNIT_CONV_CURRENT     = 2
    C_UNIT_CONV_FORCE       = 3
    C_UNIT_CONV_POWER       = 4
    C_UNIT_CONV_MASS        = 5
    C_UNIT_CONV_TIME        = 6
    C_UNIT_CONV_TEMPERATURE = 7


## -------------------------------------------------------------------------------------------------
    def __init__(self,
                 p_name:str,
                 p_id:int=None,
                 p_type:int=None,
                 p_unit_in:str=None,
                 p_unit_out:str=None,
                 p_logging=Log.C_LOG_ALL,
                 **p_args) -> None:

        super().__init__(p_name=p_name,
                         p_id=p_id,
                         p_type=p_type,
                         p_unit_in=p_unit_in,
                         p_unit_out=p_unit_out,
                         p_dt=0,
                         p_logging=p_logging,
                         p_args=None)


## -------------------------------------------------------------------------------------------------
    def __call__(self, p_input:float, p_range=None) -> float:
        """
        This method provides a functionality to call the unit converter by giving an input value.

        Parameters
        ----------
        p_input : float
            input value.
        p_range :
            not necessary for a unit converter. Default: None.

        Returns
        -------
        output : float
            output value.
        """
        if self.get_type() == self.C_UNIT_CONV_TEMPERATURE:
            output = self._temperature(p_input)
        else:
            output = self._scalar_conversion(p_input)
        
        return output


## -------------------------------------------------------------------------------------------------
    def _set_function_parameters(self, p_args=None) -> bool:
        """
        This method provides a functionality to set the parameters of the unit converter.

        Parameters
        ----------
        p_args :
            not necessary for a unit converter. Default: None.

        Returns
        -------
        bool
            true means initialization is successful.
        """
        if self.get_type() == self.C_UNIT_CONV_LENGTH:
            self.units = {
                'fm'        : 1000000000000000,
                'pm'        : 1000000000000,
                'nm'        : 1000000000,
                'um'        : 1000000,
                'mm'        : 1000,
                'cm'        : 100,
                'm'         : 1.0,
                'dam'       : 0.1,
                'hm'        : 0.01,
                'km'        : 0.001,
                'Mm'        : 0.000001,
                'Gm'        : 0.000000001,
                'Tm'        : 0.000000000001,
                'Pm'        : 0.000000000000001,
                'inch'      : 39.3701,
                'ft'        : 3.28084,
                'yd'        : 1.09361,
                'mi'        : 0.000621371,
                'nautMi'    : 1.0/1852.0,
                'lightYear' : 1.0/(9.4607304725808*(10**15))
                }
        
        elif self.get_type() == self.C_UNIT_CONV_PRESSURE:
            self.units = {
                'Pa'    : 100000.0,
                'hPa'   : 1000.0,
                'kPa'   : 100.0,
                'MPa'   : 0.1,
                'bar'   : 1.0,
                'mbar'  : 1000.0,
                'ubar'  : 1000000.0,
                'kgcm2' : 1.01972,
                'atm'   : 0.986923,
                'mmHg'  : 750.062,
                'mmH2O' : 10197.162129779,
                'mH2O'  : 10.197162129779,
                'ftH2O' : 33.455256555148,
                'inH2O' : 401.865,
                'inHg'  : 29.53,
                'psi'   : 14.5038
                }
        
        elif self.get_type() == self.C_UNIT_CONV_CURRENT:
            self.units = {
                'fA'    : 1000000000000000,
                'pA'    : 1000000000000,
                'nA'    : 1000000000,
                'uA'    : 1000000,
                'mA'    : 1000,
                'cA'    : 100,
                'A'     : 1.0,
                'daA'   : 0.1,
                'hA'    : 0.01,
                'kA'    : 0.001,
                'MA'    : 0.000001,
                'GA'    : 0.000000001,
                'TA'    : 0.000000000001,
                'PA'    : 0.000000000000001,
                }
        
        elif self.get_type() == self.C_UNIT_CONV_FORCE:
            self.units = {
                'fN'        : 1000000000000000,
                'pN'        : 1000000000000,
                'nN'        : 1000000000,
                'uN'        : 1000000,
                'mN'        : 1000,
                'cN'        : 100,
                'N'         : 1.0,
                'daN'       : 0.1,
                'hN'        : 0.01,
                'kN'        : 0.001,
                'MN'        : 0.000001,
                'GN'        : 0.000000001,
                'TN'        : 0.000000000001,
                'PN'        : 0.000000000000001,
                'shortTonF' : 1.124045e-4,
                'longTonF'  : 1.003611e-4,
                'kipf'      : 2.248089e-4,
                'lbf'       : 2.248089431e-1,
                'ozf'       : 3.5969430896,
                'pdf'       : 7.2330138512,
                'gf'        : 1.019716213e+2,
                'kgf'       : 1.019716213e-1,
                'dyn'       : 1e+5,
                'J/m'       : 1.0,
                'J/cm'      : 100.0
                }
        
        elif self.get_type() == self.C_UNIT_CONV_POWER:
            self.units = {
                'fW'                : 1000000000000000*1e3,
                'pW'                : 1000000000000*1e3,
                'nW'                : 1000000000*1e3,
                'uW'                : 1000000*1e3,
                'mW'                : 1000*1e3,
                'cW'                : 100*1e3,
                'W'                 : 1.0*1e3,
                'daW'               : 0.1*1e3,
                'hW'                : 0.01*1e3,
                'kW'                : 0.001*1e3,
                'MW'                : 0.000001*1e3,
                'GW'                : 0.000000001*1e3,
                'TW'                : 0.000000000001*1e3,
                'PW'                : 0.000000000000001*1e3,
                'BTU/hr'            : 3412.14,
                'BTU/min'           : 56.869,
                'BTU/sec'           : 0.94781666666,
                'cal/sec'           : 238.85,
                'cal/min'           : 238.85*60,
                'cal/hr'            : 238.85*60*60,
                'erg/sec'           : 10e9,
                'erg/min'           : 10e9*60,
                'erg/hr'            : 10e9*60*60,
                'ftlb/sec'          : 737.56,
                'kCal/sec'          : 0.24,
                'kCal/min'          : 0.24*60,
                'kCal/hr'           : 0.24*60*60,
                'VA'                : 1e3,
                'metric_ton_ref'    : 0.259,
                'US_ton_ref'        : 0.2843451361,
                'J/sec'             : 1000.0,
                'J/min'             : 1000.0*60,
                'J/hr'              : 1000.0*60*60,
                'kgf-m/sec'         : 101.97162129779,
                'hp_mech'           : 1.3410220888,
                'hp_ele'            : 1.3404825737,
                'hp_metric'         : 1.359621617304
                }
        
        elif self.get_type() == self.C_UNIT_CONV_MASS:
            self.units = {
                'fg'        : 1000000000000000*1e3,
                'pg'        : 1000000000000*1e3,
                'ng'        : 1000000000*1e3,
                'ug'        : 1000000*1e3,
                'mg'        : 1000*1e3,
                'cg'        : 100*1e3,
                'g'         : 1.0*1e3,
                'dag'       : 0.1*1e3,
                'hg'        : 0.01*1e3,
                'kg'        : 0.001*1e3,
                'Mg'        : 0.000001*1e3,
                'Gg'        : 0.000000001*1e3,
                'Tg'        : 0.000000000001*1e3,
                'Pg'        : 0.000000000000001*1e3,
                'metricTon' : 1.0/1000.0,
                'shortTon'  : 1.0/907.185,
                'longTon'   : 1.0/1016.047,
                'slug'      : 1.0/14.5939029,
                'lb'        : 2.2046226218,
                'oz'        : 35.274,
                'grain'     : 2.2046226218*7000.0
                }
        
        elif self.get_type() == self.C_UNIT_CONV_TIME:
            self.units = {
                'fs'        : 1000000000000000,
                'ps'        : 1000000000000,
                'ns'        : 1000000000,
                'us'        : 1000000,
                'ms'        : 1000,
                'cs'        : 100,
                's'         : 1.0,
                'das'       : 0.1,
                'hs'        : 0.01,
                'ks'        : 0.001,
                'Ms'        : 0.000001,
                'Gs'        : 0.000000001,
                'Ts'        : 0.000000000001,
                'Ps'        : 0.000000000000001,
                'min'       : 1.0/60.0,
                'hr'        : 1.0/60.0/60.0,
                'day'       : 1.0/60.0/60.0/24.0
                }
        
        elif self.get_type() == self.C_UNIT_CONV_TEMPERATURE:
            self.units = {
                'K' : 'Kelvin',
                'R' : 'Rankine',
                'F' : 'Fahrenheit',
                'C' : 'Celcius',
                }
                
        if self.units.get(self._unit_in) is not None and self.units.get(self._unit_out) is not None:
            return True
        else:
            raise NotImplementedError('The input and/or output units do not exist!')


## -------------------------------------------------------------------------------------------------
    def _scalar_conversion(self, p_input:float) -> float:
        """
        This method provides a scalar conversion functionality.

        Parameters
        ----------
        p_input : float
            input value.

        Returns
        -------
        float
            output value.
        """
        return (p_input/self.units[self._unit_in]*self.units[self._unit_out])


## -------------------------------------------------------------------------------------------------
    def _temperature(self, p_input:float) -> float:
        """
        This method provides a temperature conversion functionality.

        Parameters
        ----------
        p_input : float
            input value.

        Returns
        -------
        float
            output value.
        """
        if self._unit_in == 'R':
            temp_K = p_input*5.0/9.0
        elif self._unit_in == 'F':
            temp_K = (p_input+459.67)/9.0*5.0
        elif self._unit_in == 'C':
            temp_K = p_input+273.15
        elif self._unit_in == 'K':
            temp_K = p_input
        
        if self._unit_out == 'R':
            return (temp_K*9.0/5.0)
        elif self._unit_out == 'F':
            return (temp_K*9.0/5.0-459.67) 
        elif self._unit_out == 'C':
            return (temp_K-273.15)
        elif self._unit_out == 'K':
            return temp_K
        