## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - The integrative middleware framework for standardized machine learning
## -- Package : mlpro.oa.streams.tasks.normalizers
## -- Module  : ztrans.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2022-12-07  1.0.0     LSB      Creation/Release
## -- 2022-12-13  1.0.1     LSB      Refactoring
## -- 2022-12-20  1.0.2     DA       Refactoring
## -- 2022-12-20  1.0.3     LSB      Bugfix
## -- 2022-12-30  1.0.4     LSB      Bugfix
## -- 2023-01-12  1.1.0     LSB      Renormalizing plot data
## -- 2023-01-24  1.1.1     LSB      Bugfix
## -- 2023-02-13  1.1.2     LSB      Bugfix: Setting the default parameter update flag ot false
## -- 2023-04-09  1.2.0     DA       Class NormalizerZTransform: new methods _adapt(), _adapt_reverse()
## -- 2023-05-03  1.2.1     DA       Bugfix in NormalizerMinMax._update_plot_2d/3d/nd
## -- 2023-05-22  1.2.2     SY       Refactoring
## -- 2024-05-22  1.3.0     DA       Refactoring and splitting
## -- 2024-05-23  1.3.1     DA       Bugfix
## -------------------------------------------------------------------------------------------------

"""
Ver. 1.3.1 (2024-05-23)

This module provides implementation for adaptive normalizers for ZTransformation
"""


from mlpro.oa.streams.basics import *
from mlpro.bf.math import normalizers as Norm



## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class NormalizerZTransform (OATask, Norm.NormalizerZTrans):
    """
    Online adaptive normalization of instances with Z-Transformation

    Parameters
    ----------
    p_name: str, optional
        Name of the task.
    p_range_max:
        Processing range of the task, default is a Thread.
    p_ada:
        True if the task has adaptivity, default is true.
    p_duplicate_data : bool
        If True, instances will be duplicated before processing. Default = False.
    p_visualize:
        True for visualization, false by default.
    p_logging:
        Logging level of the task. Default is Log.C_LOG_ALL
    p_kwargs:
        Additional task parameters
    """

    C_NAME = 'Normalizer Z Transform'

## -------------------------------------------------------------------------------------------------
    def __init__(self, p_name: str = None,
                 p_range_max=StreamTask.C_RANGE_THREAD,
                 p_ada: bool = True,
                 p_duplicate_data : bool = False,
                 p_visualize = False,
                 p_logging=Log.C_LOG_ALL,
                 **p_kwargs):

        OATask.__init__(self,
            p_name=p_name,
            p_range_max=p_range_max,
            p_ada=p_ada,
            p_duplicate_data = p_duplicate_data,
            p_visualize = p_visualize,
            p_logging=p_logging,
            **p_kwargs)

        Norm.NormalizerZTrans.__init__(self)


## -------------------------------------------------------------------------------------------------
    def _run(self, p_inst : InstDict):
        """
        Custom method to for run Z-transform task for normalizing new instances and denormalizing deleted instances.

        Parameters
        ----------
        p_inst : InstDict
            Stream instances to be processed

        """

        # 1 Online update of transformation parameters
        self.adapt( p_inst = p_inst )

        # 2 Z-transformation of stream instances
        for inst_id, (inst_type, inst) in sorted(p_inst.items()):

            feature_data = inst.get_feature_data()

            if self._param is None:
                if inst_type == InstTypeNew:
                    self.update_parameters( p_data_new = feature_data )
                else:
                    self.update_parameters( p_data_del = feature_data )

            feature_data.set_values( p_values = self.normalize(feature_data).get_values() )


## -------------------------------------------------------------------------------------------------
    def _adapt(self, p_inst_new : Instance) -> bool:
        """
        Custom method to for adapting of Z-transform parameters on new instances.

        Parameters
        ----------
        p_inst_new: Instance
            Instance to be adapted on.

        Returns
        -------
        adapted : bool
            Returns True, if task has adapted.

        """

        self.update_parameters( p_data_new = p_inst_new.get_feature_data() )
        return True


## -------------------------------------------------------------------------------------------------
    def _adapt_reverse(self, p_inst_del:Instance) -> bool:
        """
        Custom method to for adapting of Z-transform parameters on deleted instances.

        Parameters
        ----------
        p_inst_del: Instance
            Instance to be adapted on.

        Returns
        -------
        adapted : bool
            Returns True, if task has adapted.

        """

        self.update_parameters( p_data_del = p_inst_del.get_feature_data() )
        return True
