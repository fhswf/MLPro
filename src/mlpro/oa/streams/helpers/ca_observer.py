## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - The integrative middleware framework for standardized machine learning
## -- Package : mlpro.oa.streams.tasks.helper
## -- Module  : ca_observer.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2025-08-27  0.1.0     DA       New class CAObserver for adaptation observation
## -------------------------------------------------------------------------------------------------

"""
Ver. 0.1.0 (2025-08-27)

This module provides the CAObserver class to be used for observation and visualization of cluster
analyzers. The class can be applied as a regular stream task. Incoming instances trigger the measurement.

"""

from mlpro.bf.various import Log
from mlpro.bf.plot import PlotSettings
from mlpro.bf.streams import InstDict, InstTypeNew, Instance, StreamTask

from mlpro.oa.streams.tasks import ClusterAnalyzer



# Export list for public API
__all__ = [ 'CAObserver' ]




## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class CAObserver (StreamTask):
    """
    This class observes the behavior of a cluster analyzer. 

    Parameters
    ----------
    p_name : str
        Optional name of the task. Default is None.
    p_range_max : int
        Maximum range of asynchonicity. See class Range. Default is Range.C_RANGE_PROCESS.
    p_duplicate_data : bool
        If True, instances will be duplicated before processing. Default = False.
    p_visualize : bool
        Boolean switch for visualisation. Default = False.
    p_logging
        Log level (see constants of class Log). Default: Log.C_LOG_ALL
    """

    C_NAME              = 'CA Observer'

    C_PLOT_ACTIVE       = True
    C_PLOT_STANDALONE   = True
    C_PLOT_VALID_VIEWS  = [ PlotSettings.C_VIEW_ND ]
    C_PLOT_DEFAULT_VIEW = PlotSettings.C_VIEW_ND

    C_PLOT_ND_YLABEL    = 'Number of clusters'

## -------------------------------------------------------------------------------------------------
    def __init__( self, 
                  p_clusterer : ClusterAnalyzer,
                  p_cluster_size_min : int = 1,
                  p_name: str = None, 
                  p_range_max = StreamTask.C_RANGE_THREAD, 
                  p_duplicate_data : bool = False,
                  p_visualize : bool = False,
                  p_logging = Log.C_LOG_ALL, 
                  **p_kwargs ):
        
        super().__init__( p_name = p_name,
                          p_range_max = p_range_max,
                          p_duplicate_data = p_duplicate_data,
                          p_visualize = p_visualize,
                          p_logging = p_logging,
                          **p_kwargs )

        self._clusterer        = p_clusterer
        self._cluster_size_min = p_cluster_size_min
        self._tstamps          = []
        self._num_clusters     = []
        self._update_plot      = False


## -------------------------------------------------------------------------------------------------
    def _run( self, p_instances : InstDict ):
        """
        Custom method that is called by method run(). 

        Parameters
        ----------
        p_instances: InstDict
            Instances to be processed.
        """

        # 1 Determine time stamp of measurement
        recent_instance : Instance = None
        for inst in p_instances.values():
            if inst[0] == InstTypeNew:
                recent_instance = inst[1]

        if recent_instance is not None: 
            tstamp = recent_instance.tstamp
        else:
            tstamp = self.get_so().tstamp


        # 2 Get number of clusters
        num_clusters = 0
        for cluster in self._clusterer.clusters.values():
            try:
                if cluster.size.value >= self._cluster_size_min:
                    num_clusters += 1
            except:
                pass


        # 3 Add measurement point to internal buffer
        try:
            self._tstamps.append(tstamp.total_seconds())
        except:
            self._tstamps.append(tstamp)

        self._num_clusters.append(num_clusters) 
        self._update_plot = True


## -------------------------------------------------------------------------------------------------
    def export_results_csv( self, 
                            p_file_name : str,
                            p_delimiter = '\t' ) -> bool:
        
        import csv
        
        try:
            with open(p_file_name, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile, delimiter=p_delimiter)
                writer.writerow(['Time Id', 'Number of Clusters'])  # Optional: Kopfzeile
                for a, b in zip(self._tstamps, self._num_clusters):
                    writer.writerow([a, b])

            print(f'File "{p_file_name}" created successfully.')
            return True
        
        except Exception as e:
            self.log( Log.C_LOG_TYPE_E, f'Error exporting results to CSV: {e}')
            return False


## -------------------------------------------------------------------------------------------------
    def _init_plot_nd(self, p_figure, p_settings: PlotSettings):
        """
        Default implementation for stream tasks. See class mlpro.bf.plot.Plottable for more
        details.
        """

        from matplotlib.ticker import MaxNLocator

        super()._init_plot_nd( p_figure=p_figure, p_settings=p_settings )

        p_settings.axes.set_xlabel(self.C_PLOT_ND_XLABEL_TIME)
        p_settings.axes.set_ylabel(self.C_PLOT_ND_YLABEL)
        p_settings.axes.grid(visible=True)
        p_settings.axes.set_autoscale_on(True)
        p_settings.axes.yaxis.set_major_locator(MaxNLocator(integer=True))        

        self._plot_num_clusters = None
        

## -------------------------------------------------------------------------------------------------
    def _update_plot_nd( self, 
                         p_settings : PlotSettings, 
                         p_instances : InstDict, 
                         **p_kwargs ) -> bool:
        """
        Time series plot of measurement data.

        Parameters
        ----------
        p_settings : PlotSettings
            Object with further plot settings.
        p_instances : InstDict
            Instances to be plotted.
        p_kwargs : dict
            Further optional plot parameters.

        Returns
        -------
        bool   
            True, if changes on the plot require a refresh of the figure. False otherwise.          
        """

        # 0 Plot update needed?
        if not self._update_plot: return False

        # 1 Late initialization of plot object
        if self._plot_num_clusters is None:
            self._plot_num_clusters, = p_settings.axes.plot([], [], lw=1 )

        # 2 Update plot data
        self._plot_num_clusters.set_data(self._tstamps, self._num_clusters)
        p_settings.axes.relim()
        p_settings.axes.autoscale_view()

        return True
