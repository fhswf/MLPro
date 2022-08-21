## -------------------------------------------------------------------------------------------------
## -- Project : FH-SWF Automation Technology - Common Code Base (CCB)
## -- Package : mlpro
## -- Module  : exceptions
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2022-08-21  0.0.0     DA       Creation
## -- 2022-08-dd  1.0.0     DA       First implementation
## -------------------------------------------------------------------------------------------------

"""
Ver. 1.0.0 (2022-08-dd)

This module provides classes for event handling. To this regard, the property class Eventmanager is
provided to add event functionality to child classes by inheritence.
"""


from mlpro.bf.various import Log




## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class Event:
    """
    Root class for events.

    Parameters
    ----------
    p_raising_object
        Reference to object that raised the event.
    """

## -------------------------------------------------------------------------------------------------
    def __init__(self, p_raising_object):
        self.raising_object = p_raising_object





## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class EventManager (Log):
    """
    This property class provides universal event management functionalities to be inherited to child
    classes.

    Parameters
    ----------
    p_logging
        Log level (see constants of class Log). Default: Log.C_LOG_ALL

    """

    C_TYPE      = 'EventManager'

## -------------------------------------------------------------------------------------------------
    def __init__(self, p_logging=Log.C_LOG_ALL):
        super().__init__(p_logging=p_logging)
        self._registered_handlers = []


## -------------------------------------------------------------------------------------------------
    def register_event_handler(self, p_event_id, p_event_handler):
        pass


## -------------------------------------------------------------------------------------------------
    def remove_event_handler(self, p_event_id, p_event_handler):
        pass


## -------------------------------------------------------------------------------------------------
    def _raise_event(self, p_event_id, p_event_object:Event):
        pass
