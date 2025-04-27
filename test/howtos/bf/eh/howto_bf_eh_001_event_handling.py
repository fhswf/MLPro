## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - A Synoptic Framework for Standardized Machine Learning Tasks
## -- Package : mlpro.bf.examples
## -- Module  : howto_bf_eh_001_event_handling.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2022-08-21  1.0.0     DA       Creation/release
## -- 2022-10-12  1.0.1     DA       Refactoring/Renaming
## -------------------------------------------------------------------------------------------------

"""
Ver. 1.0.1 (2022-10-12)

This module demonstrates the use of MLPro's event handling as a property in own classes. To this
regard, a demo class MyMainClass is set up that inherits event functionalities from MLPro's class 
EventManager. Furthermore an own sample event handler class MyHandlerClass is implemented. 


You will learn:

1) how to implement an own class with event management functionality

2) how to implement an own event handler class

3) how to register event handlers

4) how to fire events

"""



from mlpro.bf.various import Log
from mlpro.bf.events import *



# 1 Definition of custom class that inherits event management functionalities from MLPro's class EventManager
class MyMainClass (EventManager):

    C_NAME          = 'My main class'

    C_EVENT_OWN     = 'MYEVENT'

    def __init__(self, p_logging=Log.C_LOG_ALL):
        super().__init__(p_logging)
 

    def do_something(self):
        eventobj = Event(p_raising_object=self, p_par1='Hello', p_par2='World!')
        self._raise_event(self.C_EVENT_OWN, eventobj)



# 2 Definition of custom event handler class
class MyHandlerClass (Log):

    C_TYPE          = 'Event handler'
    C_NAME          = 'My handler'

    def myhandler(self, p_event_id, p_event_object:Event):
        self.log(Log.C_LOG_TYPE_I, 'Received event id', p_event_id)
        self.log(Log.C_LOG_TYPE_I, 'Event data:', p_event_object.get_data())



# 3 Instantiation of own event handler and main class as event manager
if __name__ == "__main__":
    # 3.1 Interactive/Demo mode
    myhandlerobj    = MyHandlerClass()
    mymainobj       = MyMainClass()

else:
    # 3.2 Unit test mode
    myhandlerobj    = MyHandlerClass(p_logging=Log.C_LOG_NOTHING)
    mymainobj       = MyMainClass(p_logging=Log.C_LOG_NOTHING)


# 4 Own event handler is registered on main class
mymainobj.register_event_handler(MyMainClass.C_EVENT_OWN, myhandlerobj.myhandler)


# 5 Event is fired
mymainobj.do_something()


# 6 Own event handler is removed from main class
mymainobj.remove_event_handler(MyMainClass.C_EVENT_OWN, myhandlerobj.myhandler)


# 7 Same event is fired again
mymainobj.do_something()