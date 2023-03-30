## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - A Synoptic Framework for Standardized Machine Learning Tasks
## -- Package : mlpro.bf
## -- Module  : events
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2022-08-21  1.0.0     DA       Creation/release
## -- 2022-10-06  1.1.0     DA       Specification of event id as string (for better observation and
## --                                to avoid collisions)
## -- 2023-03-25  1.1.1     DA       Class EventManager: correction in constructor
## -------------------------------------------------------------------------------------------------

"""
Ver. 1.1.1 (2023-03-25)

This module provides classes for event handling. To this regard, the property class Eventmanager is
provided to add event functionality to child classes by inheritence.
"""


from mlpro.bf.various import Log
from mlpro.bf.exceptions import *




## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class Event:
    """
    Root class for events. It is ready to use and transfers the raising object and further key/value
    data to the event handler.

    Parameters
    ----------
    p_raising_object
        Reference to object that raised the event.
    **p_kwargs 
        List of named parameters
    """

## -------------------------------------------------------------------------------------------------
    def __init__(self, p_raising_object, **p_kwargs):
        self._raising_object = p_raising_object
        self._data           = p_kwargs


## -------------------------------------------------------------------------------------------------
    def get_raising_object(self):
        return self._raising_object


## -------------------------------------------------------------------------------------------------
    def get_data(self):
        return self._data





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
        Log.__init__(self, p_logging=p_logging)
        self._registered_handlers = {}


## -------------------------------------------------------------------------------------------------
    def register_event_handler(self, p_event_id:str, p_event_handler):
        """
        Registers an event handler. 

        Parameters 
        ----------
        p_event_id : str
            Unique event id
        p_event_handler
            Reference to an event handler method with parameters p_event_id and p_event_object:Event
        """

        try:
            self._registered_handlers[p_event_id].append(p_event_handler)
        except:
            self._registered_handlers[p_event_id] = [ p_event_handler ]


## -------------------------------------------------------------------------------------------------
    def remove_event_handler(self, p_event_id:str, p_event_handler):
        """
        Removes an already registered event handler.

        Parameters 
        ----------
        p_event_id 
            Unique event id
        p_event_handler
            Reference to an event handler method.
        """

        try:
            self._registered_handlers[p_event_id].remove(p_event_handler)
        except:
            pass


## -------------------------------------------------------------------------------------------------
    def _raise_event(self, p_event_id:str, p_event_object:Event):
        """
        Raises an event and calls all registered handlers. To be used inside an event manager class.

        Parameters
        ----------
        p_event_id : str
            Unique event id
        p_event_object : Event
            Event object with further context informations
        """

        # 0 Intro
        self.log(Log.C_LOG_TYPE_S, 'Event "' + p_event_id + '" fired')

        # 1 Get list of registered handlers for given event id
        try:
            handlers = self._registered_handlers[p_event_id]
        except:
            handlers = []

        if len(handlers) == 0:
            self.log(Log.C_LOG_TYPE_I, 'No handlers registered for event "' + p_event_id + '"')
            return

        # 2 Call all registered handlers
        for i, handler in enumerate(handlers):
            try:
                self.log(Log.C_LOG_TYPE_I, 'Calling handler', i)
                handler( p_event_id=p_event_id, p_event_object=p_event_object )
            except:
                self.log(Log.C_LOG_TYPE_E, 'Handler not compatible! Check your code!')
                raise Error