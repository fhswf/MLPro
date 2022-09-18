## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - A Synoptic Framework for Standardized Machine Learning Tasks
## -- Package : mlpro.bf
## -- Module  : events
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2022-08-21  1.0.0     DA       Creation/release
## -- 2022-09-18  1.1.0     MRD      Add EventTimer
## -------------------------------------------------------------------------------------------------

"""
Ver. 1.1.0 (2022-09-18)

This module provides classes for event handling. To this regard, the property class Eventmanager is
provided to add event functionality to child classes by inheritence.
"""


import threading
import time
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
class EventTimer(threading.Thread):
    """
    Calling callback based on timer.

    Parameters
    ----------
    p_period
        Period time per call.
    p_callback
        Callback function to be called
    p_once
        Called only once
    """

    def __init__(self, p_period, p_callback, p_once=False):
        threading.Thread.__init__(self)
        self._period   = p_period
        self._callback = p_callback
        self._once     = p_once
        self._shutdown = False
        self.daemon = True
        self.start()

    def shutdown(self):
        """
        Stop firing callbacks.
        """
        self._shutdown = True
        
    def run(self):
        while not self._shutdown:
            time.sleep(self._period)
            self._callback()
            if self._once:
                break


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
        self._registered_handlers = {}


## -------------------------------------------------------------------------------------------------
    def register_event_handler(self, p_event_id, p_event_handler):
        """
        Registers an event handler. 

        Parameters 
        ----------
        p_event_id 
            Unique event id
        p_event_handler
            Reference to an event handler method with parameters p_event_id and p_event_object:Event
        """

        try:
            self._registered_handlers[p_event_id].append(p_event_handler)
        except:
            self._registered_handlers[p_event_id] = [ p_event_handler ]


## -------------------------------------------------------------------------------------------------
    def remove_event_handler(self, p_event_id, p_event_handler):
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
    def _raise_event(self, p_event_id, p_event_object:Event):
        """
        Raises an event and calls all registered handlers. To be used inside an event manager class.

        Parameters
        ----------
        p_event_id 
            Unique event id
        p_event_object : Event
            Event object with further context informations
        """

        # 0 Intro
        self.log(Log.C_LOG_TYPE_I, 'Event', str(p_event_id), 'fired')

        # 1 Get list of registered handlers for given event id
        try:
            handlers = self._registered_handlers[p_event_id]
        except:
            handlers = []

        if len(handlers) == 0:
            self.log(Log.C_LOG_TYPE_I, 'No handlers registered for event', str(p_event_id))
            return

        # 2 Call all registered handlers
        for i, handler in enumerate(handlers):
            try:
                self.log(Log.C_LOG_TYPE_I, 'Calling handler', i)
                handler( p_event_id=p_event_id, p_event_object=p_event_object )
            except:
                self.log(Log.C_LOG_TYPE_E, 'Handler not compatible! Check your code!')
                raise Error