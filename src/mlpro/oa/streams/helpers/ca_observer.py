## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - The integrative middleware framework for standardized machine learning
## -- Package : mlpro.oa.streams.tasks.helper
## -- Module  : ca_observer.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2025-08-27  0.1.0     DA       New class CAObserver for adaptation observation
## -- 2025-09-15  0.2.0     DA       Redesign and extension
## -- 2025-09-22  0.3.0     DA       - Parameters to turn on/off each sub-plot
## --                                - Parameter for minimum cluster size
## --                                - Bugfix in centroid accuracy calculation
## -- 2025-09-23  0.3.1     DA       Bugfixes
## -------------------------------------------------------------------------------------------------

"""
Ver. 0.3.1 (2025-09-23)

This module provides the CAObserver class to be used for observation and visualization of cluster
analyzers. The class can be applied as a regular stream task. Incoming instances trigger the measurement.

"""

from matplotlib.ticker import MaxNLocator
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
    p_cluster_min_size : int = 1
        Minimum size of clusters to be considered as valid (default = 1).
    p_sw_acc_clusters : bool = False
        Boolean switch for activating the accuracy measurement of the number of clusters. Default is False.
    p_sw_acc_centroids : bool = False
        Boolean switch for activating the accuracy measurement of the centroids. Default is False.
    p_sw_acc_total : bool = True
        Boolean switch for activating the total accuracy measurement. Default is True.
    p_sw_err_clusters : bool = True
        Boolean switch for activating the error measurement of the number of clusters. Default is True.
    p_sw_err_centroids : bool = True
        Boolean switch for activating the error measurement of the centroids. Default is True.
    p_sw_num_clusters_exp : bool = False
        Boolean switch for activating the plot of the number of expected clusters. Default is False.
    p_sw_num_clusters_obs : bool = False
        Boolean switch for activating the plot of the number of observed clusters. Default is False.
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

    C_NAME                           = 'CA Observer'

    C_PLOT_ACTIVE                    = True
    C_PLOT_STANDALONE                = True
    C_PLOT_VALID_VIEWS               = [ PlotSettings.C_VIEW_ND ]
    C_PLOT_DEFAULT_VIEW              = PlotSettings.C_VIEW_ND
  
    C_PLOT_ND_YLABEL_LEFT            = 'Accuracies/Errors [%]'
    C_PLOT_ND_YLABEL_RIGHT           = 'Number of clusters'
    C_PLOT_ND_LABEL_ACC_CLUSTERS     = 'Acc. clusters'
    C_PLOT_ND_LABEL_ACC_CENTROIDS    = 'Acc. centroids'
    C_PLOT_ND_LABEL_ACC_TOTAL        = 'Acc. total'
    C_PLOT_ND_LABEL_ERR_CLUSTERS     = 'Err. clusters'
    C_PLOT_ND_LABEL_ERR_CENTROIDS    = 'Err. centroids'
    C_PLOT_ND_LABEL_NUM_CLUSTERS_EXP = 'Num. clusters (exp.)'
    C_PLOT_ND_LABEL_NUM_CLUSTERS_OBS = 'Num. clusters (obs.)'

## -------------------------------------------------------------------------------------------------
    def __init__( self, 
                  p_clusterer : ClusterAnalyzer,
                  p_cluster_statistics : ClusterStatistics,
                  p_cluster_min_size : int = 1,
                  p_sw_acc_clusters : bool = False,
                  p_sw_acc_centroids : bool = False,
                  p_sw_acc_total : bool = True,
                  p_sw_err_clusters : bool = True,
                  p_sw_err_centroids : bool = True,
                  p_sw_num_clusters_exp : bool = False, 
                  p_sw_num_clusters_obs : bool = False, 
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

        self._clusterer           = p_clusterer
        self._cluster_stats       = p_cluster_statistics
        self._cluster_min_size    = p_cluster_min_size
        self._sw_acc_clusters     = p_sw_acc_clusters
        self._sw_acc_centroids    = p_sw_acc_centroids
        self._sw_acc_total        = p_sw_acc_total
        self._sw_err_clusters     = p_sw_err_clusters
        self._sw_err_centroids    = p_sw_err_centroids
        self._sw_num_clusters_exp = p_sw_num_clusters_exp
        self._sw_num_clusters_obs = p_sw_num_clusters_obs

        # Observation buffers
        self._tstamps          = []
        self._acc_clusters     = []
        self._acc_centroids    = []
        self._acc_total        = []
        self._err_clusters     = []
        self._err_centroids    = []
        self._num_clusters_exp = []
        self._num_clusters_obs = []

        # Plotting
        self._plot_ax2_max_y   = 0
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
                if cluster.size.value >= self._cluster_min_size:
                    num_clusters_obs += 1
            except:
                pass

        self._num_clusters_obs.append(num_clusters_obs) 


        # 4 Update accuracies and errors

        # 4.1 Stage 1: Accuracy/error of number of clusters
        acc_clusters = 100 * max((1 - abs(num_clusters_obs - num_clusters_exp) / num_clusters_exp), 0)
        err_clusters = 100.0 - acc_clusters
        self._acc_clusters.append(acc_clusters)
        self._err_clusters.append(err_clusters)

        # 4.2 Stage 2: Accuracy/error of centroids
        if num_clusters_obs == num_clusters_exp:
            acc_centroids = self._get_centroid_accuracy()
            err_centroids = 100.0 - acc_centroids
        else:
            acc_centroids = 0.0
            err_centroids = 100.0
        self._acc_centroids.append(acc_centroids)
        self._err_centroids.append(err_centroids)

        # 4.3 Total accuracy
        acc_total = (acc_clusters + acc_centroids) / 2
        self._acc_total.append(acc_total)

        # 4.4 Max y-value for axis 2 (number of clusters)
        if self.get_visualization() and self._ax2_active:
            self._plot_ax2_max_y = max( self._plot_ax2_max_y, num_clusters_exp, num_clusters_obs )


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
        actual   = [c.centroid.value for c in self._clusterer.clusters.values() if c.size.value is not None and c.size.value > 0]


        # 2 Check cluster number consistency
        if len(expected) != len(actual) or not expected:
            return 0.0

        expected = np.array(expected)
        actual   = np.array(actual)


        # # 3 Load feature ranges
        # ranges = np.array(self._cluster_stats.feature_boundaries)
        # ranges[ranges == 0] = 1  # avoid division by zero

        # 3 Load feature ranges
        boundaries = np.array(self._cluster_stats.feature_boundaries)  # shape (d, 2)
        ranges = boundaries[:, 1] - boundaries[:, 0]                   # shape (d,)
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
                writer.writerow(['Time Id', 'Number of Clusters (expected)', 'Number of Clusters (observed)', 'Error clusters[%]', 'Error centroids[%]', 'Accuracy clusters[%]', 'Accuracy centroids[%]', 'Accuracy total[%]'])  
                for a, b, c, d, e, f, g, h in zip( self._tstamps, 
                                             self._num_clusters_exp, 
                                             self._num_clusters_obs, 
                                             self._err_clusters, 
                                             self._err_centroids,
                                             self._acc_clusters, 
                                             self._acc_centroids, 
                                             self._acc_total ):
                    writer.writerow([a, b, c, d, e, f, g, h])

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

        # from matplotlib.ticker import MaxNLocator

        super()._init_plot_nd( p_figure=p_figure, p_settings=p_settings )

        p_settings.axes.set_xlabel(self.C_PLOT_ND_XLABEL_TIME)
        p_settings.axes.set_ylabel(self.C_PLOT_ND_YLABEL)
        p_settings.axes.grid(visible=True)
        p_settings.axes.set_ylim(0, 100)
        p_settings.axes.set_ylabel(self.C_PLOT_ND_YLABEL_LEFT)

        if self._sw_num_clusters_exp or self._sw_num_clusters_obs:
            p_settings.axes2 = p_settings.axes.twinx()
            p_settings.axes2.set_ylabel(self.C_PLOT_ND_YLABEL_RIGHT)
            self._ax2_active = True
        else:
            self._ax2_active = False

        self._plot_acc_clusters     = None
        self._plot_acc_centroids    = None
        self._plot_acc_total        = None
        self._plot_err_clusters     = None
        self._plot_err_centroids    = None
        self._plot_num_clusters_exp = None
        self._plot_num_clusters_obs = None

        self._first_plot            = True


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
        if self._first_plot:

            ax1_labels = []
            ax1_plots = []

            if self._sw_acc_centroids:
                self._plot_acc_centroids, = p_settings.axes.plot([], [], lw=1, color='lightgreen', label=self.C_PLOT_ND_LABEL_ACC_CENTROIDS)
                ax1_labels.append(self.C_PLOT_ND_LABEL_ACC_CENTROIDS)
                ax1_plots.append(self._plot_acc_centroids)

            if self._sw_acc_clusters:
                self._plot_acc_clusters,  = p_settings.axes.plot([], [], lw=1, color='green', label=self.C_PLOT_ND_LABEL_ACC_CLUSTERS)
                ax1_labels.append(self.C_PLOT_ND_LABEL_ACC_CLUSTERS)
                ax1_plots.append(self._plot_acc_clusters)

            if self._sw_acc_total:
                self._plot_acc_total,     = p_settings.axes.plot([], [], lw=2, color='darkgreen', label=self.C_PLOT_ND_LABEL_ACC_TOTAL)
                ax1_labels.append(self.C_PLOT_ND_LABEL_ACC_TOTAL)
                ax1_plots.append(self._plot_acc_total)

            if self._sw_err_clusters:
                self._plot_err_clusters,  = p_settings.axes.plot([], [], lw=1, color='red', label=self.C_PLOT_ND_LABEL_ERR_CLUSTERS)
                ax1_labels.append(self.C_PLOT_ND_LABEL_ERR_CLUSTERS)
                ax1_plots.append(self._plot_err_clusters)

            if self._sw_err_centroids:
                self._plot_err_centroids, = p_settings.axes.plot([], [], lw=1, color='orange', label=self.C_PLOT_ND_LABEL_ERR_CENTROIDS)
                ax1_labels.append(self.C_PLOT_ND_LABEL_ERR_CENTROIDS)
                ax1_plots.append(self._plot_err_centroids)

            if self._sw_num_clusters_exp:
                self._plot_num_clusters_exp, = p_settings.axes2.plot([], [], lw=1, ls='--', color='lightblue', label=self.C_PLOT_ND_LABEL_NUM_CLUSTERS_EXP)
                ax1_labels.append(self.C_PLOT_ND_LABEL_NUM_CLUSTERS_EXP)
                ax1_plots.append(self._plot_num_clusters_exp)

            if self._sw_num_clusters_obs:
                self._plot_num_clusters_obs, = p_settings.axes2.plot([], [], lw=1, ls='--', color='blue', label=self.C_PLOT_ND_LABEL_NUM_CLUSTERS_OBS)
                ax1_labels.append(self.C_PLOT_ND_LABEL_NUM_CLUSTERS_OBS)
                ax1_plots.append(self._plot_num_clusters_obs)

            # Legend 1: Accuracy plots
            leg1 = p_settings.axes.legend( 
                ax1_plots,
                ax1_labels,
                loc="upper left",
                bbox_to_anchor=(0.0, -0.10),   # linksbündig, unterhalb der Achse
                ncol=3,
                frameon=False
            )

            p_settings.axes.add_artist(leg1)
            self._figure.subplots_adjust(bottom=0.2)
            p_settings.axes.autoscale_view(scalex=True, scaley=False)

            if self._ax2_active:
                p_settings.axes2.autoscale_view(scalex=False, scaley=True)

            self._first_plot = False


        # 2 Update plot data
        if self._sw_acc_clusters:
            self._plot_acc_clusters.set_data(self._tstamps, self._acc_clusters)

        if self._sw_acc_centroids:
            self._plot_acc_centroids.set_data(self._tstamps, self._acc_centroids)

        if self._sw_acc_total:
            self._plot_acc_total.set_data(self._tstamps, self._acc_total)

        if self._sw_err_clusters:
            self._plot_err_clusters.set_data(self._tstamps, self._err_clusters)

        if self._sw_err_centroids:
            self._plot_err_centroids.set_data(self._tstamps, self._err_centroids)

        if self._ax2_active:
            if self._sw_num_clusters_exp:
                self._plot_num_clusters_exp.set_data(self._tstamps, self._num_clusters_exp)

            if self._sw_num_clusters_obs:
                self._plot_num_clusters_obs.set_data(self._tstamps, self._num_clusters_obs)

            p_settings.axes2.set_ylim(0, self._plot_ax2_max_y + 1)


        # 3 Update x scaling
        p_settings.axes.set_xlim(self._tstamps[0], self._tstamps[-1])        

        return True
