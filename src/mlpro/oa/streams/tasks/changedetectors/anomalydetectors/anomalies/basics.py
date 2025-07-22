## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - The integrative middleware framework for standardized machine learning
## -- Package : mlpro.oa.tasks.anomalydetectors.anomalies
## -- Module  : basics.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2023-06-08  0.0.0     SK       Creation
## -- 2023-09-12  1.0.0     SK       Release
## -- 2023-11-21  1.0.1     SK       Time Stamp update
## -- 2024-02-25  1.1.0     SK       Visualisation update
## -- 2024-04-10  1.2.0     DA/SK    Refactoring
## -- 2024-04-11  1.3.0     DA       Class Anomaly: extensions on methods update_plot_*
## -- 2024-05-07  1.3.1     SK       Bug fix related to p_instances
## -- 2024-05-09  1.3.2     DA       Bugfix in method Anomaly._update_plot()
## -- 2024-05-22  1.4.0     SK       Refactoring
## -- 2025-02-12  1.4.1     DA       Code reduction
## -- 2025-02-18  2.0.0     DA       Class Anomaly:
## --                                - refactoring and simplification
## --                                - new attribute event_id
## --                                - new parent Renormalizable
## -- 2025-05-28  2.1.0     DA/DS    Class Anomaly: new parent Change
## -- 2025-07-18  2.2.0     DA       Refactoring
## -------------------------------------------------------------------------------------------------

"""
Ver. 2.2.0 (2025-07-18)

This module provides a template class for anomalies to be used in anomaly detection algorithms.
"""


from mlpro.oa.streams.tasks.changedetectors import Change



# Export list for public API
__all__ = [ 'Anomaly' ]




## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class Anomaly (Change):
    """
    This is the base class for anomaly events raised by the anomaly detectors. See parent class 
    Change for more details.
    """

    pass