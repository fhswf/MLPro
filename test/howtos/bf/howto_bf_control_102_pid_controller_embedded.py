## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - The integrative middleware framework for standardized machine learning
## -- Package : test/howtos/bf
## -- Module  : howto_bf_control_102_pid_controller_embedded.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2024-09-01  0.0.0     DA       Creation
## -------------------------------------------------------------------------------------------------

"""
Ver. 0.0.0 (2024-09-01)

This module demonstrates ...

You will learn:

1) How to ...

2) How to ...

3) How to ...

"""


from mlpro.bf.various import Log



## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class MyDemo (Log):
    """
    This class demonstrates how to ...
    """

    # needed for proper logging (see class mlpro.bf.various.Log)
    C_TYPE      = 'Demo'
    C_NAME      = 'Parallel Algorithm'

## -------------------------------------------------------------------------------------------------
    def __init__( self, 
                  p_logging=Log.C_LOG_ALL ):

        super().__init__( p_logging=p_logging )


## -------------------------------------------------------------------------------------------------
    def execute(self):
        # Log something
        self.log(Log.C_LOG_TYPE_I, 'Here we go...')






# 1 Preparation of demo/unit test mode
if __name__ == '__main__':
    # 1.1 Parameters for demo mode
    cycle_limit = 200
    logging     = Log.C_LOG_ALL
    visualize   = True
  
else:
    # 1.2 Parameters for internal unit test
    cycle_limit = 2
    logging     = Log.C_LOG_NOTHING
    visualize   = False



# 2 Instantiate the demo objects
demo = MyDemo( p_logging = logging )



# 3 Demo actions
demo.execute()
