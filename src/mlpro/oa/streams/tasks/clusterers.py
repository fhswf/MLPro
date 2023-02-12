## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - A Synoptic Framework for Standardized Machine Learning Tasks
## -- Package : mlpro.oa.tasks.boundarydetectors
## -- Module  : anomalydetectors.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2023-01-24  0.0.0     DA       Creation
## -------------------------------------------------------------------------------------------------

"""
Ver. 0.0.0 (2023-01-24)

This module provides templates for cluster analysis to be used in the context of online adaptivity.
"""

from mlpro.oa.streams import *
import random





## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class Cluster (Model):
    """
    This is the base class for a multivariate cluster. 

    Parameters
    ----------
    p_visualize : bool
        Boolean switch for visualisation. Default = False.
    """

    C_TYPE                  = 'Cluster'
    C_NAME                  = '????'

    C_PLOT_ACTIVE           = True
    C_PLOT_STANDALONE       = False
    C_PLOT_VALID_VIEWS      = [ PlotSettings.C_VIEW_2D, 
                                PlotSettings.C_VIEW_3D, 
                                PlotSettings.C_VIEW_ND ]
    C_PLOT_DEFAULT_VIEW     = PlotSettings.C_VIEW_ND

## -------------------------------------------------------------------------------------------------
    def __init__( self,  
                  p_id,
                  p_center = None,
                  p_sample_buffer_size : int = 0,
                  p_sample_buffer_step : int = 10,
                  p_sample_buffer_rnd : bool = True,
                  p_visualize : bool = False,
                  p_logging = Log.C_LOG_ALL,
                  **p_kwargs ):

        Log.__init__(self, p_logging=p_logging)
        Plottable.__init__(self, p_visualize=p_visualize)

        self._id                 = p_id
        self._center             = p_center
        self._boundaries         = []
        self._num_instances      = 0
        self._sample_buffer      = {}
        self._sample_buffer_size = p_sample_buffer_size
        self._sample_buffer_step = p_sample_buffer_step
        self._sample_buffer_rnd  = p_sample_buffer_rnd
        self._sample_buffer_skip = 0
        self._kwargs             = p_kwargs.copy()


## -------------------------------------------------------------------------------------------------
    def get_id(self):
        return self._id


## -------------------------------------------------------------------------------------------------
    def set_id(self, p_id):
        self._id = p_id


## -------------------------------------------------------------------------------------------------
    def get_sample_set(self) -> list[Instance]:
        return self._sample_buffer.values()


## -------------------------------------------------------------------------------------------------
    def set_random_seed(p_seed=None):
        random.seed(p_seed)


## -------------------------------------------------------------------------------------------------
    def _update_sample_buffer(self, p_inst_new: list[Instance], p_inst_del: list[Instance] ):
        
        # 1 Remove obsolete instances from the buffer
        for inst in p_inst_del:
            try:
                del self._sample_buffer[inst.get_id()]
            except:
                pass


        # 2 Add new instances to the buffer
        for inst in p_inst_new:
            if self._sample_buffer_skip == 0:
                if self._sample_buffer_rnd:
                    self._sample_buffer_skip = random.randint(1,self._sample_buffer_step)
                else:
                    self._sample_buffer_skip = self._sample_buffer_step

                self._sample_buffer[inst.get_id()] = inst

                if len(self._sample_buffer) > self._sample_buffer_size:
                    del self._sample_buffer[ next(iter(self._sample_buffer.keys())) ]

            else:
                self._sample_buffer_skip -= 1


## -------------------------------------------------------------------------------------------------
    def update(self, p_inst_new: list[Instance], p_inst_del: list[Instance]) -> list:
        """
        Updates the cluster based on instances to be added or removed. It calls custom method _update()
        and optionally maintains a set of sample instances.

        Parameters
        ----------
        p_inst_new : list[Instance]
            Optional list of new stream instances to be processed. If None, the list of the shared object
            is used instead. Default = None.
        p_inst_del : list[Instance]
            List of obsolete stream instances to be removed. If None, the list of the shared object
            is used instead. Default = None.

        Returns
        -------
        children : list
            List of new child clusters spawned by the custom implementation.
        """
        
        self.log(Log.C_LOG_TYPE_I, 'Updating cluster from instances...')
        children = []
        self._update( p_inst_new=p_inst_new, p_inst_del=p_inst_del, p_children=children )
        l = len(children)
        if l > 0:
            self.log(Log.C_LOG_TYPE_S, 'Cluster has spawned', l, 'children')
            if self._sample_buffer_size > 0:
                self._sample_buffer.clear()
                self.log(Log.C_LOG_TYPE_S, 'Sample buffer cleared')
        elif self._sample_buffer_size > 0:
            self._update_sample_buffer( p_inst_new=p_inst_new, p_inst_del=p_inst_del )

        return children


## -------------------------------------------------------------------------------------------------
    def _update( self, 
                 p_inst_new : list, 
                 p_inst_del : list, 
                 p_children : list ):
        """
        Custom method that updates a cluster based on instances to be added or removed. Depending on
        the specific implementation the creation ("spawning") of new clusters is possible. In that
        case just add new child clusters to the list parameter p_children.

        Parameters
        ----------
        p_inst_new : list[Instance]
            Optional list of new stream instances to be processed. If None, the list of the shared object
            is used instead. Default = None.
        p_inst_del : list[Instance]
            List of obsolete stream instances to be removed. If None, the list of the shared object
            is used instead. Default = None.
        p_children : list
            List of new clusters
        """

        pass


## -------------------------------------------------------------------------------------------------
    def _init_plot_2d(self, p_figure: Figure, p_settings: PlotSettings):
        raise NotImplementedError


## -------------------------------------------------------------------------------------------------
    def _init_plot_3d(self, p_figure: Figure, p_settings: PlotSettings):
        raise NotImplementedError


## -------------------------------------------------------------------------------------------------
    def _init_plot_nd(self, p_figure: Figure, p_settings: PlotSettings):
        raise NotImplementedError


## -------------------------------------------------------------------------------------------------
    def _update_plot_2d(self, p_settings: PlotSettings, **p_kwargs):
        raise NotImplementedError


## -------------------------------------------------------------------------------------------------
    def _update_plot_3d(self, p_settings: PlotSettings, **p_kwargs):
        raise NotImplementedError


## -------------------------------------------------------------------------------------------------
    def _update_plot_nd(self, p_settings: PlotSettings, **p_kwargs):
        raise NotImplementedError





## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class ClusterAnalyzer (OATask):
    """
    This is the base class for multivariate online cluster analysis. It raises an event when a cluster
    was added or removed.
    """

    C_NAME                  = 'Cluster Analyzer'

    C_EVENT_CLUSTER_ADDED   = 'CLUSTER_ADDED'
    C_EVENT_CLUSTER_REMOVED = 'CLUSTER_REMOVED'

    C_PLOT_ACTIVE           = True
    C_PLOT_STANDALONE       = False

    def __init__( self, 
                p_name: str = None, p_range_max=StreamTask.C_RANGE_THREAD, p_ada: bool = True, p_duplicate_data: bool = False, p_visualize: bool = False, p_logging=Log.C_LOG_ALL, **p_kwargs):
        super().__init__(p_name, p_range_max, p_ada, p_duplicate_data, p_visualize, p_logging, **p_kwargs)


## -------------------------------------------------------------------------------------------------
    def set_random_seed(self, p_seed=None):
        return super().set_random_seed(p_seed)


## -------------------------------------------------------------------------------------------------
    def _run(self, p_inst_new: list, p_inst_del: list):
        raise NotImplementedError