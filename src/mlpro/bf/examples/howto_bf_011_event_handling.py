## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - A Synoptic Framework for Standardized Machine Learning Tasks
## -- Package : mlpro
## -- Module  : howto_bf_001_logging.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2022-08-21  0.0.0     DA       Creation
## -- 2022-08-dd  1.0.0     DA       First implementation
## -------------------------------------------------------------------------------------------------

"""
Ver. 1.0.0 (2022-08-dd)

This module demonstrates the use of MLPro's event handling as a property in own classes. To this
regard, a demo class MyClass is set up that inherits event functionalities from MLPro's class EventManager.
"""



from mlpro.bf.various import Log
from mlpro.bf.events import *




# 1 Custom class that inherits event management functionalities from MLPro's class EventManager
class MyClass (EventManager):

    C_EVENT_OWN     = 0

    def __init__(self, p_logging=Log.C_LOG_ALL):
        super().__init__(p_logging)
 

    def do_something(self):
        pass




# 2 Custom event handler class
class MyHandlerClass (Log):

    def myhandler(self, p_event_id, p_event_object):
        pass





if __name__ == "__main__":
    pass


else:
    pass