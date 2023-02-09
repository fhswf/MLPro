## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - A Synoptic Framework for Standardized Machine Learning Tasks
## -- Package : mlpro.bf.systems
## -- Module  : adaptive_systems.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2023-02-09  0.0.0     LSB      Creation
## -------------------------------------------------------------------------------------------------

"""
Ver. 0.0.0 (2023-02-09)

This module provides models and templates for adaptive state based systems.
"""



from mlpro.bf.systems import *
from mlpro.bf.ml import Model, Mode
from mlpro.bf.math import *
from typing import Union
from mlpro.rl.models_env_ada import AFctSTrans, AFctBroken, AFctSuccess





## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class ASystem(System, Model):
    """
    This is a template class for Adaptive Systems

    Parameters
    ----------
    p_mode
    p_latency
    p_fct_strans
    p_fct_success
    p_fct_broken
    p_visualize
    p_logging
    """


    ## -------------------------------------------------------------------------------------------------
    def __init__(self,
                 p_mode = Mode.C_MODE_SIM,
                 p_latency = None,
                 p_fct_strans: Union[FctSTrans, AFctSTrans] = None,
                 p_fct_success: Union[FctSuccess, AFctSuccess] = None,
                 p_fct_broken: Union[FctBroken, AFctBroken] = None,
                 p_visualize: bool = False,
                 p_logging =Log.C_LOG_ALL):


        System.__init__(self,
                        p_mode = p_mode,
                        p_latency = p_latency,
                        p_fct_strans = p_fct_strans,
                        p_fct_success = p_fct_success,
                        p_fct_broken = p_fct_broken,
                        p_visualize = p_visualize,
                        p_logging = p_logging)


        self._fct_strans  = p_fct_strans
        self._fct_broken  = p_fct_broken
        self._fct_success = p_fct_success


## -------------------------------------------------------------------------------------------------
    def _set_adapted(self, p_adapted:bool):
        """

        Parameters
        ----------
        p_adapted

        Returns
        -------

        """

        Model._set_adapted(self, p_adapted=p_adapted)


## -------------------------------------------------------------------------------------------------
    def switch_adaptivity(self, p_ada:bool):
        """

        Parameters
        ----------
        p_ada

        Returns
        -------

        """
        try: self._fct_strans.switch_adaptivity(p_ada=p_ada)
        except: pass

        try: self._fct_broken.switch_adaptivity(p_ada=p_ada)
        except:pass

        try: self._fct_success.switch_adaptivity(p_ada=p_ada)
        except: pass


## -------------------------------------------------------------------------------------------------
    def _adapt(self, **p_kwargs) -> bool:
        """

        Parameters
        ----------
        p_kwargs

        Returns
        -------

        """

        try: self._fct_strans.adapt()
        except: pass

        try: self._fct_broken.adapt()
        except: pass

        try: self._fct_success.adapt()
        except: pass
