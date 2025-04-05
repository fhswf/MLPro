## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - The integrative middleware framework for standardized machine learning
## -- Package : mlpro.bf
## -- Module  : howto_bf_006_time_stamps.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2024-05-21  1.0.0     DA       Creation/Release
## -------------------------------------------------------------------------------------------------

"""
Ver. 1.0.0 (2024-05-21)

This module demonstrates the basic handling of time stamps.

"""


from mlpro.bf.various import TStamp, Log
from datetime import datetime, timedelta
from pathlib import Path



# 1 Preparation of demo/unit test mode
if __name__ == '__main__':
    # 1.1 Parameters for demo mode
    logging     = Log.C_LOG_ALL
 
else:
    # 1.2 Parameters for internal unit test
    logging     = Log.C_LOG_NOTHING


# 2 Preparation of loggin
log = Log(p_logging=logging)
log.C_TYPE = 'Demo'
log.C_NAME = 'Time Stamps'


# 3 Integer time stamps
t1 = TStamp(10)
t2 = TStamp(20)
log.log(Log.C_LOG_TYPE_I, 'Integer time stamps: ', t1.tstamp, t2.tstamp, t2.tstamp - t1.tstamp)


# 4 Float time stamps
t1 = TStamp(10.5)
t2 = TStamp(20.7)
log.log(Log.C_LOG_TYPE_I, 'Float time stamps: ', t1.tstamp, t2.tstamp, t2.tstamp - t1.tstamp)


# 5 Absolute real time stamps
t1 = TStamp(datetime.now())
t2 = TStamp(datetime.now() + timedelta(seconds = 1))
log.log(Log.C_LOG_TYPE_I, 'Absolute real time stamps: ', t1.tstamp, t2.tstamp, t2.tstamp - t1.tstamp)


# 6 Relative real time stamps
t1 = TStamp(timedelta(seconds = 10.5))
t2 = TStamp(timedelta(seconds = 11.5))
log.log(Log.C_LOG_TYPE_I, 'Relative real time stamps: ', t1.tstamp, t2.tstamp, t2.tstamp - t1.tstamp)


