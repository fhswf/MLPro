## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - A Synoptic Framework for Standardized Machine Learning Tasks
## -- Package : mlpro
## -- Module  : howto_bf_001_logging.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2021-10-07  1.0.0     DA       Creation
## -- 2021-11-03  1.1.0     DA       Introduction of new log type C_LOG_TYPE_S for success messages
## -- 2021-11-13  1.1.1     DA       Refactoring
## -------------------------------------------------------------------------------------------------

"""
Ver. 1.1.1 (2021-11-13)

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
        self.log(self.C_LOG_TYPE_I, 'But don\'t worry. Everything is fine. It\'s just a demo:)')
        self.log(self.C_LOG_TYPE_S, 'This method terminated successfully!\n')




if __name__ == "__main__":
    
    # 2 Log everything inside your class... 
    print('\n--\n-- Example 1: By default everything is logged...\n--\n')
    mc = MyClass(p_logging=Log.C_LOG_ALL)
    mc.my_method()


    # 3 Log nothing inside your class
    print('\n--\n-- Example 2: Now logging is switched off...\n--\n')
    mc.switch_logging(Log.C_LOG_NOTHING)
    mc.my_method()


    # 4 Log warnings and errors only
    print('\n--\n-- Example 3: Only warnings and errors are logged...\n--\n')
    mc.switch_logging(Log.C_LOG_WE)
    mc.my_method()

    # 5 Log errors only
    print('\n--\n-- Example 4: Only errors are logged...\n--\n')
    mc.switch_logging(Log.C_LOG_E)
    mc.my_method()

    # 6 Log everything again
    print('\n--\n-- Example 5: Everything is logged again...\n--\n')
    mc.switch_logging(Log.C_LOG_ALL)
    mc.my_method()
