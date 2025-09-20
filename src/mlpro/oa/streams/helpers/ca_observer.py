## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - The integrative middleware framework for standardized machine learning
## -- Package : mlpro.oa.streams.tasks.helper
## -- Module  : ca_observer.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2025-08-27  0.1.0     DA       New class CAObserver for adaptation observation
## -- 2025-09-15  0.2.0     DA       Redesign and extension
## -------------------------------------------------------------------------------------------------

"""
Ver. 0.2.0 (2025-09-15)

This module provides the CAObserver class to be used for observation and visualization of cluster
analyzers. The class can be applied as a regular stream task. Incoming instances trigger the measurement.

"""

import numpy as np
from scipy.optimize import linear_sum_assignment

from mlpro.bf.various import Log
from mlpro.bf.plot import PlotSettings
from mlpro.bf.streams import InstDict, InstTypeNew, Instance, StreamTask
from mlpro.bf.streams.streams import ClusterStatistics

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
    p_clusterer : ClusterAnalyzer
        The cluster analyzer to be observed.
    p_cluster_statistics : ClusterStatistics
        The cluster statistics object providing information about the expected clusters.
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

    C_NAME                            = 'CA Observer'

    C_PLOT_ACTIVE                     = True
    C_PLOT_STANDALONE                 = True
    C_PLOT_VALID_VIEWS                = [ PlotSettings.C_VIEW_ND ]
    C_PLOT_DEFAULT_VIEW               = PlotSettings.C_VIEW_ND
  
    C_PLOT_ND_YLABEL_LEFT             = 'Accuracies [%]'
    C_PLOT_ND_YLABEL_RIGHT            = 'Number of clusters'
    C_PLOT_ND_YLABEL_ACC_CLUSTERS     = 'Accuracy clusters [%]'
    C_PLOT_ND_YLABEL_ACC_CENTROIDS    = 'Accuracy centroids [%]'
    C_PLOT_ND_YLABEL_ACC_TOTAL        = 'Accuracy total [%]'
    C_PLOT_ND_YLABEL_NUM_CLUSTERS_EXP = 'Number of clusters (expected)'
    C_PLOT_ND_YLABEL_NUM_CLUSTERS_OBS = 'Number of clusters (observed)'

    C_PLOT_COLORS                     = [ 'blue', 
                                          'orange', 
                                          'green', 
                                          'red', 
                                          'purple']

## -------------------------------------------------------------------------------------------------
    def __init__( self, 
                  p_clusterer : ClusterAnalyzer,
                  p_cluster_statistics : ClusterStatistics,
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
        self._cluster_stats    = p_cluster_statistics

        # Observation buffers
        self._tstamps          = []
        self._num_clusters_exp = []
        self._num_clusters_obs = []
        self._acc_clusters     = []
        self._acc_centroids    = []
        self._acc_total        = []

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


        # 2 Add measurement point to internal buffer
        try:
            self._tstamps.append(tstamp.total_seconds())
        except:
            self._tstamps.append(tstamp)


        # 3 Update number of expected and observed clusters
        num_clusters_exp = self._cluster_stats.num_clusters
        self._num_clusters_exp.append(num_clusters_exp)

        num_clusters_obs = 0
        for cluster in self._clusterer.clusters.values():
            try:
                if cluster.size.value >= 1:
                    num_clusters_obs += 1
            except:
                pass

        self._num_clusters_obs.append(num_clusters_obs) 


        # 4 Update accuracies

        # 4.1 Stage 1: Accuracy of number of clusters
        acc_clusters = 100 * max((1 - abs(num_clusters_obs - num_clusters_exp) / num_clusters_exp), 0)
        self._acc_clusters.append(acc_clusters)

        # 4.2 Stage 2: Accuracy of centroids
        if num_clusters_obs == num_clusters_exp:
            acc_centroids = self._get_centroid_accuracy()
        else:
            acc_centroids = 0.0
        self._acc_centroids.append(acc_centroids)

        # 4.3 Total accuracy
        acc_total = (acc_clusters + acc_centroids) / 2
        self._acc_total.append(acc_total)


        # 5 Trigger plot update
        self._update_plot = True


## -------------------------------------------------------------------------------------------------
    def _get_centroid_accuracy(self) -> float:
        """
        Computes the accuracy between expected and actual cluster centroids.

        The evaluation is two-staged:
        1) The method first checks whether the number of expected and actual clusters matches.
        If not, the accuracy is defined as 0.0 and no further comparison is made.
        2) If the numbers match, the method computes a normalized distance matrix between
        all pairs of expected and actual centroids. The normalization is performed with
        respect to the known feature boundaries provided in
        `self._cluster_stats.feature_boundaries`.

        To resolve the unknown mapping between expected and actual clusters, the Hungarian
        algorithm (`scipy.optimize.linear_sum_assignment`) is applied to find the optimal
        assignment that minimizes the total centroid deviation.

        The final accuracy is derived as:
            accuracy = 100 * (1 - mean_normalized_rmse)

        Returns
        -------
        float
            Accuracy in percent within [0, 100].
        """

        # 1 Get expected and actual centroids
        expected = [c.center for c in self._cluster_stats.clusters.values()]
        actual   = [c.centroid.value for c in self._clusterer.clusters.values() if c.size.value > 0]


        # 2 Check cluster number consistency
        if len(expected) != len(actual) or not expected:
            return 0.0

        expected = np.array(expected)
        actual   = np.array(actual)


        # 3 Load feature ranges
        ranges = np.array(self._cluster_stats.feature_boundaries)
        ranges[ranges == 0] = 1  # avoid division by zero


        # 4 Build distance matrix (normalized RMSE for each pair)
        n = len(expected)
        dist_matrix = np.zeros((n, n))
        for i, exp in enumerate(expected):
            for j, act in enumerate(actual):
                mse_norm = np.mean(((exp - act) / ranges) ** 2)
                rmse_norm = np.sqrt(mse_norm)
                dist_matrix[i, j] = rmse_norm


        # 5 Find optimal assignment (Hungarian algorithm)
        row_ind, col_ind = linear_sum_assignment(dist_matrix)


        # 6 Compute mean normalized RMSE across all assignments
        mean_rmse = dist_matrix[row_ind, col_ind].mean()


        # 7 Derive accuracy in percent
        accuracy = 100 * (1 - mean_rmse)
        return max(0.0, accuracy)


## -------------------------------------------------------------------------------------------------
    def export_results_csv( self, 
                            p_file_name : str,
                            p_delimiter = '\t' ) -> bool:
        
        import csv
        
        try:
            with open(p_file_name, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile, delimiter=p_delimiter)
                writer.writerow(['Time Id', 'Number of Clusters (expected)', 'Number of Clusters (observed)', 'Accuracy clusters[%]', 'Accuracy centroids[%]', 'Accuracy total[%]'])  
                for a, b, c, d, e, f in zip(self._tstamps, self._num_clusters_exp, self._num_clusters_obs, self._acc_clusters, self._acc_centroids, self._acc_total):
                    writer.writerow([a, b, c, d, e, f])

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
        p_settings.axes.set_ylim(0, 100)
        p_settings.axes.set_ylabel(self.C_PLOT_ND_YLABEL_LEFT)

        # self._figure.subplots_adjust(bottom=0.3)

        p_settings.axes2 = p_settings.axes.twinx()
        p_settings.axes2.set_ylabel(self.C_PLOT_ND_YLABEL_RIGHT)
        # p_settings.axes2.yaxis.set_major_locator(MaxNLocator(integer=True))      

        self._plot_acc_clusters     = None
        self._plot_acc_centroids    = None
        self._plot_acc_total        = None
        self._plot_num_clusters_exp = None
        self._plot_num_clusters_obs = None


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
        if self._plot_acc_clusters is None:

            colors = self.C_PLOT_COLORS

            self._plot_acc_clusters,     = p_settings.axes.plot([], [], lw=1, color=colors[0], label=self.C_PLOT_ND_YLABEL_ACC_CLUSTERS)
            self._plot_acc_centroids,    = p_settings.axes.plot([], [], lw=1, color=colors[1], label=self.C_PLOT_ND_YLABEL_ACC_CENTROIDS)
            self._plot_acc_total,        = p_settings.axes.plot([], [], lw=2, color=colors[2], label=self.C_PLOT_ND_YLABEL_ACC_TOTAL)

            self._plot_num_clusters_exp, = p_settings.axes2.plot([], [], lw=1, ls='--', color=colors[3], label=self.C_PLOT_ND_YLABEL_NUM_CLUSTERS_EXP)
            self._plot_num_clusters_obs, = p_settings.axes2.plot([], [], lw=1, ls='--', color=colors[4], label=self.C_PLOT_ND_YLABEL_NUM_CLUSTERS_OBS)

            # Legend 1: Accuracy plots
            leg1 = p_settings.axes.legend(
                [self._plot_acc_clusters, self._plot_acc_centroids, self._plot_acc_total],
                [self._plot_acc_clusters.get_label(),
                self._plot_acc_centroids.get_label(),
                self._plot_acc_total.get_label()],
                loc="upper left",
                bbox_to_anchor=(0.0, -0.10),   # linksbündig, unterhalb der Achse
                ncol=3,
                frameon=False
            )

            # Legend 2: Cluster count plots
            leg2 = p_settings.axes.legend(
                [self._plot_num_clusters_exp, self._plot_num_clusters_obs],
                [self._plot_num_clusters_exp.get_label(),
                self._plot_num_clusters_obs.get_label()],
                loc="upper left",
                bbox_to_anchor=(0.0, -0.15),   # noch etwas weiter unten
                ncol=2,
                frameon=False
            )

            p_settings.axes.add_artist(leg1)
            self._figure.tight_layout()


        # 2 Update plot data
        self._plot_acc_clusters.set_data(self._tstamps, self._acc_clusters)
        self._plot_acc_centroids.set_data(self._tstamps, self._acc_centroids)
        self._plot_acc_total.set_data(self._tstamps, self._acc_total)   
        self._plot_num_clusters_exp.set_data(self._tstamps, self._num_clusters_exp)
        self._plot_num_clusters_obs.set_data(self._tstamps, self._num_clusters_obs)

        p_settings.axes2.relim()
        p_settings.axes2.autoscale_view()

        return True
