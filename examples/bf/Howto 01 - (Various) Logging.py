## -------------------------------------------------------------------------------------------------
## -- Project : FH-SWF Automation Technology - Common Code Base (CCB)
## -- Package : mlpro
## -- Module  : Howto 01 - (Various) Logging
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2021-10-07  1.0.0     DA       Creation
## -------------------------------------------------------------------------------------------------

"""
Ver. 1.0.0 (2021-10-07)

This module demonstrates the Log class functionality.
"""


from mlpro.bf.various import Log




# 1 Reuse logging property in your own class by inheriting from class Log
class MyClass(Log):

    # These constants are inherited fromm class Log and will be logged in every log line...
    C_TYPE  = 'Demo class'      
    C_NAME  = 'MyClass'

    def __init__(self, p_logging=True):
        # The constructor of class Log initializes the internal logging and writes the first line...
        super().__init__(p_logging=p_logging)


    def my_method(self):
        # The log types I/E/W are also inherited from class Log...
        self.log(self.C_LOG_TYPE_I, 'Let me tell you what\'s going on...')
        self.log(self.C_LOG_TYPE_W, 'Something is weird...')
        self.log(self.C_LOG_TYPE_E, 'And here something failed...')
        self.log(self.C_LOG_TYPE_I, 'But don\'t worry. Everything is fine. It\'s just a demo:)\n\n')





# 2 Log everything inside your class... 
mc = MyClass(p_logging=True)
mc.my_method()


# 3 Log nothing inside your class
mc.switch_logging(False)
mc.my_method()


# 4 Log warnings and errors only
mc.switch_logging(True)
mc.set_log_level(Log.C_LOG_TYPE_W)
mc.my_method()

# 5 Log errors only
mc.set_log_level(Log.C_LOG_TYPE_E)
mc.my_method()

# 6 Log everything again
mc.set_log_level(Log.C_LOG_TYPE_I)
mc.my_method()

