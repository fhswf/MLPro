## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - The integrative middleware framework for standardized machine learning
## -- Package : mlpro.bf
## -- Module  : events
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2022-08-21  1.0.0     DA       Creation/release
## -- 2022-10-06  1.1.0     DA       Specification of event id as string (for better observation and
## --                                to avoid collisions)
## -- 2023-03-25  1.1.1     DA       Class EventManager: correction in constructor
## -- 2023-11-17  1.2.0     DA       Class Event: new time stamp functionality
## -- 2023-11-18  1.2.1     DA       Class Event: time stamp is set to now() if not provided
## -- 2024-05-23  1.3.0     DA       Method EventManger._raise_event(): reduction to TypeError   
## -- 2025-05-27  1.4.0     DA       Class Event: new parent class KWArgs
## -- 2025-07-18  1.5.0     DA       Refactoring
## -------------------------------------------------------------------------------------------------

"""
Ver. 1.5.0 (2025-07-18)

This module provides classes for event handling. To this regard, the property class Eventmanager is
provided to add event functionality to child classes by inheritence.
"""

from datetime import datetime
from mlpro.bf.various import Log, TStamp, TStampType, KWArgs
from mlpro.bf.exceptions import *



# Export list for public API
__all__ = [ 'Event',
            'EventManager' ]




## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class Event (TStamp, KWArgs):
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
    def __init__(self, p_raising_object, p_tstamp:TStampType = None, **p_kwargs):
        self._raising_object = p_raising_object

        if p_tstamp is None:
            TStamp.__init__(self, p_tstamp = datetime.now())
        else:
            TStamp.__init__(self, p_tstamp = p_tstamp)

        KWArgs.__init__(self, **p_kwargs)


## -------------------------------------------------------------------------------------------------
    def get_raising_object(self):
        return self._raising_object


## -------------------------------------------------------------------------------------------------
    def get_data(self):
        return self.kwargs





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
            except TypeError:
                self.log(Log.C_LOG_TYPE_E, 'Handler not compatible! Check your code!')
                raise TypeError