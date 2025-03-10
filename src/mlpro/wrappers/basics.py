## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - The integrative middleware framework for standardized machine learning
## -- Package : mlpro.wrappers
## -- Module  : basics.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2021-04-16  0.0.0     DA       Creation
## -- 2021-05-29  1.0.0     DA       Release of first version
## -- 2021-06-16  1.1.0     SY       Adding the first version of data storing,
## --                                data plotting, and data saving classes
## -- 2021-06-17  1.2.0     DA       New abstract classes Loadable, Saveable
## -- 2021-06-21  1.3.0     SY       Add extensions in classes Loadable,
## --                                Saveable, DataPlotting & DataStoring.
## -- 2021-07-01  1.4.0     SY       Extend save/load functionalities
## -- 2021-08-20  1.5.0     DA       Added property class Plottable
## -- 2021-08-28  1.5.1     DA       Added constant C_VAR0 to class DataStoring
## -- 2021-09-11  1.5.0     MRD      Change Header information to match our new library name
## -- 2021-10-06  1.5.2     DA       Moved class DataStoring to new module mlpro.bf.data.py and
## --                                classes DataPlotting, Plottable to new module mlpro.bf.plot.py
## -- 2021-10-07  1.6.0     DA       Class Log: 
## --                                - colored text depending on log type 
## --                                - new method set_log_level()
## -- 2021-10-25  1.7.0     SY       Add new class ScientificObject
## -- 2021-11-03  1.7.1     DA       Class Log: new type C_LOG_TYPE_SUCCESS for success messages 
## -- 2021-11-15  1.7.2     DA       Class Log: 
## --                                - method set_log_level() removed
## --                                - parameter p_logging is the new log level now
## --                                Class Saveable: new constant C_SUFFIX
## -- 2021-12-07  1.7.3     SY       Add a new attribute in ScientificObject
## -- 2021-12-31  1.7.4     DA       Class Log: udpated docstrings
## -- 2022-07-21  1.8.0     DA       New class Wrapper
## -- 2023-01-14  1.8.1     MRD      Save installed version
## -- 2025-03-10  1.9.0     DA       Replaced outdated package pkg_resources with importlib
## -------------------------------------------------------------------------------------------------

"""
Ver. 1.9.0 (2025-03-10)

This module provides model classes for wrappers in the MLPro project.
"""

from importlib.metadata import version as pkg_version

from mlpro.bf.exceptions import *
from mlpro.bf.various import Log




## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class Wrapper (Log):
    """
    Root class for all MLPro wrapper classes. Please specify the wrapped package in attibute 
    C_WRAPPED_PACKAGE and an optional minimum version in attribute C_MINIMUM_VERSION.

    Parameters
    ----------
    p_logging
        Log level (see constants of class Log). Default = Log.C_LOG_ALL.
    """

    C_TYPE                      = 'Wrapper'
    C_WRAPPED_PACKAGE : str     = None
    C_MINIMUM_VERSION : str     = None

## -------------------------------------------------------------------------------------------------
    def __init__(self, p_logging=Log.C_LOG_ALL):
        Log.__init__(self, p_logging=p_logging)

        if self.C_WRAPPED_PACKAGE is None:
            raise Error('Please specify the wrapped package')

        try:
            version = pkg_version(self.C_WRAPPED_PACKAGE)
            self.log(Log.C_LOG_TYPE_I, 'Wrapped package ' + self.C_WRAPPED_PACKAGE + ' installed in version ' + version)
            self.installed_version = version

        except:
            raise Error('Package ' + self.C_WRAPPED_PACKAGE + ' not installed')

        if self.C_MINIMUM_VERSION is not None:
            ver_actual = version.split('.')
            ver_setpoint = self.C_MINIMUM_VERSION.split('.')

            for i, val_setpoint in enumerate(ver_setpoint):
                val_actual = ver_actual[i]

                if val_actual < val_setpoint:
                    self.log(Log.C_LOG_TYPE_W, 'Minimum version ' + self.C_MINIMUM_VERSION + ' of package ' + self.C_WRAPPED_PACKAGE + ' fallen below. Please update.')
                    break

                elif val_actual > val_setpoint:
                    break
 