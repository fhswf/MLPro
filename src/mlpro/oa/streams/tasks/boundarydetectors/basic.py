## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - The integrative middleware framework for standardized machine learning
## -- Package : mlpro.oa.streams.tasks.boundarydetectors
## -- Module  : basic.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2022-10-17  0.0.0     LSB      Creation
## -- 2022-10-21  1.0.0     LSB      Release
## -- 2022-12-07  1.0.1     LSB      Refactoring for scaler factor
## -- 2022-12-08  1.0.2     LSB      Compatibiltty for instance and Element objects
## -- 2022-12-12  1.0.3     DA       Corrected signature of method _adapt_on_event()
## -- 2022-12-13  1.0.4     LSB      Refactoring
## -- 2022-12-16  1.0.5     LSB      Refactoring for get_related_set method
## -- 2022-12-20  1.0.6     LSB      Bug Fixes
## -- 2022-12-20  1.1.0     LSB      ND Visualization
## -- 2022-12-20  1.1.1     LSB      Refactoring visualization
## -- 2022-12-20  1.1.2     LSB      Bug Fix
## -- 2022-12-28  1.1.3     DA       Class BoundaryDetector: 
## --                                - added constant C_PLOT_VALID_VIEWS
## --                                - removed methods init_plot_2d/3d, update_plot_2d/3d
## -- 2022-12-30  1.1.4     DA       Removed the plot title
## -- 2023-02-02  1.1.5     DA       Method BoundaryDetector._init_plot_2D: removed figure creation
## -- 2023-02-13  1.1.6     SY       Bug Fix: Solving issue, when the first data is lower than 1
## -- 2023-02-13  1.1.7     LSB      Bug Fix: Setting initial boundaries to [0,0]
## -- 2023-04-09  1.2.0     DA       Refactoring of method BoundaryDetector._adapt()
## -- 2023-05-02  1.2.1     DA       Class BoundaryDetector
## --                                - constructor: removed parameter p_window
## --                                - method _adapt(): removed unnecessary code
## -- 2023-05-20  1.2.2     DA       Method BoundaryDetector._adapt_on_event: refactoring, corrections
## -- 2023-05-21  1.2.3     LSB      Bug Fix : p_scaler shall be generated as a vertical array
## -- 2023-11-19  1.2.4     DA       Bugfix in method BoundaryDetector._adapt(): scaler management
## -- 2024-05-12  1.3.0     DA       Removed the scaler functionality from BoundaryDetector
## -- 2024-05-22  1.4.0     DA       Refactoring and correction in BoundaryDetector._adapt()
## -- 2024-10-29  1.5.0     DA       - Refactoring
## --                                - Pseudo-implementation of BoundaryDetector._adapt_reverse()
## -- 2024-11-05  1.5.1     DA       Bugfix in method BoundaryDetector._upate_plot_nd()
## -- 2024-12-11  1.5.2     DA       Pseudo classes if matplotlib is not installed
## -- 2025-06-04  2.0.0     DA       Refactoring:
## --                                - new parent BoundaryProvider
## --                                - replaced event-oriented adaptation by reverse adaptation
## --                                  with callback to external boundary provider 
## --                                  (e.g. a sliding window)
## -- 2025-06-06  2.1.0     Da       - Refactoring: p_inst -> p_instances
## --                                - BoundaryDetector._update_plot_nd() reworked
## -- 2025-06-08  2.2.0     DA       Refactoring of methods BoundaryDetector._update_plot_nd(): new 
## --                                return param
## -- 2025-06-25  2.3.0     DA       Reduced the boundary bar's width
## -------------------------------------------------------------------------------------------------

"""
Ver. 2.3.0 (2025-06-25)

This module provides a basic implementation of a boundary detector.

"""

from typing import Union

try:
    import matplotlib.colors
    from matplotlib.figure import Figure
    from matplotlib.patches import Rectangle
except:
    class Figure : pass

import numpy as np

from mlpro.bf.various import Log
from mlpro.bf.plot import PlotSettings
from mlpro.bf.math import Set
from mlpro.bf.math.statistics import Boundaries, BoundarySide, BoundaryProvider
from mlpro.bf.streams import Instance, InstDict, InstTypeNew, InstTypeDel
from mlpro.oa.streams.basics import OAStreamTask, OAStreamAdaptationType



# Export list for public API
__all__ = [ 'BoundaryDetector' ]




## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class BoundaryDetector (OAStreamTask, BoundaryProvider):
    """
    This class provides the functionality of boundary observation of incoming instances. 

    Parameters
    ----------
    p_name: str, Optional.
        Name of the task.
    p_range_max
        Processing range of the task. Default is thread.
    p_ada: bool
        True if the task has adaptivity. Default is True.
    p_duplicate_data : bool
        If True, instances will be duplicated before processing. Default = False.
    p_visualize: bool
        True to turn on the visualization.
    p_logging
        Logging level for the task, default is Log all.
    p_boundary_provider : BoundaryProvider = None
        Optional boundary provider, used for service call on reverse adaptation.
    """

    C_TYPE                  = 'Boundary Detector'

    C_PLOT_ND_XLABEL        = 'Features'
    C_PLOT_ND_YLABEL        = 'Boundaries'

    C_PLOT_STANDALONE       = True
    C_PLOT_VALID_VIEWS      = [ PlotSettings.C_VIEW_ND ]
    C_PLOT_DEFAULT_VIEW     = PlotSettings.C_VIEW_ND

## -------------------------------------------------------------------------------------------------
    def __init__( self,
                  p_name:str = None,
                  p_range_max = OAStreamTask.C_RANGE_THREAD,
                  p_ada : bool = True,
                  p_duplicate_data : bool = False,
                  p_visualize : bool = False,
                  p_logging = Log.C_LOG_ALL,
                  p_boundary_provider : BoundaryProvider = None,
                  **p_kwargs ):

        super().__init__( p_name = p_name,
                          p_range_max = p_range_max,
                          p_ada = p_ada,
                          p_duplicate_data = p_duplicate_data,
                          p_visualize = p_visualize,
                          p_logging = p_logging,
                          **p_kwargs )

        self._related_set: Set        = None
        self._boundary_provider       = p_boundary_provider
        self._boundaries : Boundaries = None
        self._boundaries_reduce       = None


## -------------------------------------------------------------------------------------------------
    def _run(self, p_instances: InstDict):
        """
        Method to run the boundary detector task

        Parameters
        ----------
        p_instances : InstDict
            Instances to be processed.
        """

        self.adapt(p_instances=p_instances)


## -------------------------------------------------------------------------------------------------
    def adapt(self, p_instances : InstDict) -> bool:

        # 0 Intro
        if not self._adaptivity: return False
        self.log(self.C_LOG_TYPE_S, 'Adaptation started')

        adapted_forward  = False
        adapted_reverse  = False
        num_inst_forward = 0
        num_inst_reverse = 0
        

        # 1 Preprocessing 
        try:
            atype_pre = self._adapt_pre()

            if atype_pre != OAStreamAdaptationType.NONE:
                if atype_pre == OAStreamAdaptationType.FORWARD: adapted_forward = True
                elif atype_pre == OAStreamAdaptationType.REVERSE: adapted_reverse = True

                self.log(self.C_LOG_TYPE_S, 'Preprocessing done')

        except NotImplementedError:
            pass


        # 2 Two-pass adaptation

        # 2.1 Pass #1: Forward adaptation -> boundary extension on new instances
        for inst_id, (inst_type, inst) in p_instances.items():

            if inst_type != InstTypeNew: continue

            self.log(self.C_LOG_TYPE_S, 'Adaptation on new instance', inst_id)
            if self._adapt( p_instance_new=inst ):
                adapted_forward   = True
                num_inst_forward += 1
                self.log(self.C_LOG_TYPE_S, 'Boundaries extended')


        # 2.2 Pass #2: Reverse adaptation -> identification of boundaries to be reducted based 
        # on obsolete instances
        for inst_id, (inst_type, inst) in p_instances.items():

            if inst_type != InstTypeDel: continue

            self.log(self.C_LOG_TYPE_S, 'Reverse adaptation on obsolete instance', inst_id)
            try:
                if self._adapt_reverse( p_instance_del=inst ):
                    adapted_reverse   = True
                    num_inst_reverse += 1
                    self.log(self.C_LOG_TYPE_S, 'Boundaries reduced')

            except NotImplementedError:
                self.log(self.C_LOG_TYPE_W, 'Reverse adaptation not implemented', inst_id)


        # 3 Postprocessing
        try:
            atype_post = self._adapt_post()

            if atype_post != OAStreamAdaptationType.NONE:
                if atype_post == OAStreamAdaptationType.FORWARD: adapted_forward = True
                elif atype_post == OAStreamAdaptationType.REVERSE: adapted_reverse = True

                self.log(self.C_LOG_TYPE_S, 'Postprocessing done')

        except NotImplementedError:
            pass


        # 4 Outro: Logging and adaptation events
        if adapted_forward or adapted_reverse:
            self.log(self.C_LOG_TYPE_S, 'Adaptation done with changes')
            tstamp = self.get_so().tstamp

            if adapted_reverse:
                self._set_adapted( p_adapted = True,
                                   p_subtype = OAStreamAdaptationType.REVERSE,
                                   p_tstamp = tstamp,
                                   p_num_inst = num_inst_reverse )

            if adapted_forward:
                self._set_adapted( p_adapted = True,
                                   p_subtype = OAStreamAdaptationType.FORWARD,
                                   p_tstamp = tstamp,
                                   p_num_inst = num_inst_forward )
                
            return True
        
        else:
            self.log(self.C_LOG_TYPE_S, 'Adaptation done without changes')
            self._set_adapted( p_adapted = False )
            return False
        

## -------------------------------------------------------------------------------------------------
    def _init_data_structures(self, p_instance : Instance ):
        feature_data            = p_instance.get_feature_data()
        num_dim                 = feature_data.get_related_set().get_num_dim()
        self._boundaries        = self._create_boundaries( p_num_dim = num_dim )
        self._boundaries_tmp    = np.ndarray( (num_dim) ) 
        self._boundaries_reduce = np.zeros((num_dim,2), dtype=bool)   
        self._related_set       = feature_data.get_related_set()


## -------------------------------------------------------------------------------------------------
    def _adapt(self, p_instance_new : Instance) -> bool:
        """
        Extends boundaries based on a new instance.

        Parameters
        ----------
        p_instance_new : Instance
            New instance to be processed.

        Returns
        -------
        bool
            True, if at least one boundary value has been extended. False otherwise.
        """

        # 0 First call: Preparation of boundary arrays
        if self._boundaries is None: self._init_data_structures( p_instance = p_instance_new )


        # 1 Extend boundaries on new instance
        adapted        = False
        feature_values = p_instance_new.get_feature_data().get_values()

        # 1.1 Upper boundaries
        np.fmax(self._boundaries[:,BoundarySide.UPPER], feature_values, out=self._boundaries_tmp)
        if np.any(self._boundaries_tmp != self._boundaries[:,BoundarySide.UPPER]):
            np.copyto( self._boundaries[:,BoundarySide.UPPER], self._boundaries_tmp )
            adapted = True

        # 1.2 Lower boundaries
        np.fmin(self._boundaries[:,BoundarySide.LOWER], feature_values, out=self._boundaries_tmp)
        if np.any(self._boundaries_tmp != self._boundaries[:,BoundarySide.LOWER]):
            np.copyto( self._boundaries[:,BoundarySide.LOWER], self._boundaries_tmp )
            adapted = True


        # 2 Return adaptation flag
        return adapted
    

## -------------------------------------------------------------------------------------------------
    def _adapt_reverse(self, p_instance_del:Instance) -> bool:
        """
        Determines the boundaries to be reduced based on an outdated instance.

        Parameters
        ----------
        p_instance_del : Instance
            Deleted instance to be processed.

        Returns
        -------
        bool
            True, if at least one boundary value needs to be reduced. False otherwise.
        """
    
        # 0 First call: Preparation of boundary arrays
        if self._boundaries is None: self._init_data_structures( p_instance = p_instance_del )  
    

        # 1 Mark boundaries for potential reduction
        feature_values = p_instance_del.get_feature_data().get_values()

        # 1.1 Upper boundaries
        np.greater_equal(
            feature_values,
            self._boundaries[:,BoundarySide.UPPER],
            out=self._boundaries_reduce[:,BoundarySide.UPPER]
        )

        # 1.2 Lower boundaries
        np.less_equal(
            feature_values,
            self._boundaries[:,BoundarySide.LOWER],
            out=self._boundaries_reduce[:,BoundarySide.LOWER]
        )


        # 2 Determine if any boundary needs to be reduced
        return np.any(self._boundaries_reduce)
    

## -------------------------------------------------------------------------------------------------
    def _adapt_post(self) -> OAStreamAdaptationType:
        """
        ...
        """

        result = OAStreamAdaptationType.NONE

        if self._boundaries_reduce is not None:
            for idx_tuple in np.argwhere(self._boundaries_reduce):
                dim, side = idx_tuple[0], idx_tuple[1]
                self._boundaries[dim,side] = self._boundary_provider.get_boundaries( p_dim = dim, p_side = side )
                result = OAStreamAdaptationType.REVERSE

            if result == OAStreamAdaptationType.REVERSE:
                self._boundaries_reduce[...] = False

        return result


## -------------------------------------------------------------------------------------------------
    def get_boundaries(self, p_dim : int = None, p_side : BoundarySide = None ) -> Union[Boundaries, float]:
        """
        Returns the current value boundaries of internally stored data. The result can be reduced
        by the optional parameters p_side, p_dim. If both parameters are specified, the result is
        a float.

        Parameters
        ----------
        p_dim : int = None
            Optionally reduces the result to a particular dimension.
        p_side : BoundarySide = None
            Optionally reduces the result to upper or lower boundaries. See class BoundarySide for
            possible values.

        Returns
        -------
        Union[Boundaries, float]
            Returns the current boundaries of the data. The return value depends on the combination 
            of the optional parameters:
            - If neither `p_side` nor `p_dim` is specified: returns the full 2×n array.
            - If only `p_side` is specified: returns a 1D array with values for all dimensions.
            - If only `p_dim` is specified: returns a 1D array with both lower and upper bounds.
            - If both `p_side` and `p_dim` are specified: returns a single float value.
        """

        try:
            i = p_dim if p_dim is not None else slice(None)
            j = p_side if p_side is not None else slice(None)
            return self._boundaries[i, j]
        
        except (IndexError, TypeError, ValueError):
            return None

    
## -------------------------------------------------------------------------------------------------
    def _init_plot_nd(self, p_figure: Figure, p_settings: PlotSettings):
        """
        Custom method to initialize plot for Boundary Detectors tasks for N-dimensional plotting.

        Parameters
        ----------
        p_figure:Figure
            Figure to host the plot
        p_settings: PlotSettings
            PlotSettings objects with specific settings for the plot

        """

        super()._init_plot_nd( p_figure = p_figure, p_settings = p_settings )

        p_settings.axes.set_xlabel(self.C_PLOT_ND_XLABEL)
        p_settings.axes.xaxis.set_label_position('top')   # Position des Labels nach oben setzen

        p_settings.axes.set_ylabel(self.C_PLOT_ND_YLABEL)
        p_settings.axes.set_axisbelow(True)

        self._plot_nd_plots = None
        self._plot_colors   = list(matplotlib.colors.TABLEAU_COLORS.values())


## --------------------------------------------------------------------------------------------------
    def _update_plot_nd(self,
                        p_settings: PlotSettings,
                        p_instances: InstDict,
                        **p_kwargs) -> bool:
        """
        N-dimensional plotting for Boundary Detector using vertical bars and mean markers.

        Parameters
        ----------
        p_settings : PlotSettings
            Plot settings object including matplotlib axes.
        p_instances : InstDict
            Optional stream instances (not used here).
        p_kwargs : dict
            Further optional parameters.

        Returns
        -------
        bool   
            True, if changes on the plot require a refresh of the figure. False otherwise.          
        """

        # 0 Abort if no boundaries are currently adapted
        if not self.get_adapted(): 
            return False


        # 1 Prepare plotting data
        dims        = self._related_set.get_dims()
        num_dims    = len(dims)
        lowers      = self._boundaries[:, BoundarySide.LOWER]
        uppers      = self._boundaries[:, BoundarySide.UPPER]
        feature_ids = np.arange(num_dims)
        labels      = [dim.get_name_short() for dim in dims]
        ax          = p_settings.axes


        # 2 First-time setup
        if self._plot_nd_plots is None:
            self._plot_nd_plots = {
                'rects_main': [],
                'rects_bg': []
            }

            # Set xlim early to get left border
            ax.set_xlim(-1, num_dims)
            x_left = ax.get_xlim()[0]

            for i in range(num_dims):
                color = self._plot_colors[i % len(self._plot_colors)]

                x_main = feature_ids[i] - 0.1
                width_bg = x_main - x_left

                # 2.1 Background rectangle ("shadow")
                rect_bg = Rectangle(
                    (x_left, lowers[i]),
                    width=width_bg,
                    height=uppers[i] - lowers[i],
                    color=color,
                    alpha=0.15,
                    linewidth=1,
                    zorder = 0
                )
                ax.add_patch(rect_bg)
                self._plot_nd_plots['rects_bg'].append(rect_bg)

                # 2.2 Main rectangle
                rect_main = Rectangle(
                    (x_main, lowers[i]),
                    width=0.2, #0.8,
                    height=uppers[i] - lowers[i],
                    facecolor=color,
                    edgecolor = 'grey',
                    linewidth = 1,
                    zorder = 1
                )
                ax.add_patch(rect_main)
                self._plot_nd_plots['rects_main'].append(rect_main)

            # 2.3 Axis setup
            ax.set_xticks(feature_ids)
            ax.set_xticklabels(labels, rotation=45, ha='right')

        else:
            # 3 Update existing plot elements
            for i in range(num_dims):
                y = lowers[i]
                h = uppers[i] - lowers[i]

                rect_bg = self._plot_nd_plots['rects_bg'][i]
                rect_bg.set_y(y)
                rect_bg.set_height(h)

                rect_main = self._plot_nd_plots['rects_main'][i]
                rect_main.set_y(y)
                rect_main.set_height(h)


        # 4 Y-Axis autoscaling
        y_min = np.min(lowers)
        y_max = np.max(uppers)
        ax.set_ylim(y_min,y_max)

        return True