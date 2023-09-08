## -------------------------------------------------------------------------------------------------
## -- Project : MLPro Marketplace
## -- Module  : update_marketplace.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2023-09-08  0.0.0     DA       Creation 
## -------------------------------------------------------------------------------------------------

"""
Ver. 0.0.0 (2023-09-08)

This standalone module collects meta data of all whitelisted GitHub repositories based on the
template repo /fhswf/MLPro-Extension.
"""


from mlpro.bf.various import Log



## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class Marketplace (Log):

    C_TYPE      = 'Marketplace'
    C_NAME      = 'MLPro'

## -------------------------------------------------------------------------------------------------
    def __init__(self, p_logging=Log.C_LOG_ALL):
        super().__init__(p_logging)




marketplace = Marketplace()