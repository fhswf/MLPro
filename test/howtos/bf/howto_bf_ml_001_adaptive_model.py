## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - A Synoptic Framework for Standardized Machine Learning Tasks
## -- Package : mlpro.bf.examples
## -- Module  : howto_bf_ml_001_adaptive_model.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2023-02-15  1.0.0     DA       Creation
## -------------------------------------------------------------------------------------------------

"""
Ver. 1.0.0 (2023-02-15)

This module demonstrates the basic properties of an adaptive model in MLPro. It also gives an overview
of all custom methods that can be used for own purposes.

You will learn:

1) How to implement a custom model

2) How to let your model adapt on a dataset

3) How to run your model as a task

"""


from mlpro.bf.ml import *



## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class MyModel (Model):

    # Please give your implementation a suitable name...
    C_NAME              = 'Demo'

    # If you want to use the visualization features you need to activate it explicitely...
    C_PLOT_ACTIVE       = True

    # Any related scientific literature? Please add the bibliographic parameters. See class
    # mlpro.bf.various.ScientificObject for further information...
    C_SCIREF_TYPE       = ScientificObject.C_SCIREF_TYPE_ARTICLE
    C_SCIREF_AUTHOR     = 'Detlef Arend, Mochammad Rizky Diprasetya, Steve Yuwono, Andreas Schwung'
    C_SCIREF_TITLE      = 'MLPro - An integrative middleware framework for standardized machine learning tasks in Python'
    C_SCIREF_JOURNAL    = 'Software Impacts'
    C_SCIREF_PUBLISHER  = 'Elsevier'
    C_SCIREF_YEAR       = '2022'
    C_SCIREF_VOLUME     = '14'
    C_SCIREF_DOI        = '10.1016/j.simpa.2022.100421'


## -------------------------------------------------------------------------------------------------
    def _init_hyperparam(self, p_hp_layers : int, p_hp_neurons_per_layer : int ):
        """
        Define and initialize the specific hyperparameters of your model here. See classes Model,
        HyperParamSpace, HyperParam, HyerParamTuple of sub-package mlpro.bf.ml for further information.

        Parameters
        ----------
        p_hp_layers : int
            Number of layers.
        p_hp_neurons_per_layer : int
            Number of neurons per layer.
        """

        # 1 Define the hyperparameter space of your model...
        self._hyperparam_space.add_dim( HyperParam( p_name_short = 'number of layers', 
                                                    p_base_set = Dimension.C_BASE_SET_N, 
                                                    p_boundaries=[1,5]) )

        self._hyperparam_space.add_dim( HyperParam( p_name_short = 'neurons per layer', 
                                                    p_base_set = Dimension.C_BASE_SET_N, 
                                                    p_boundaries=[1,20]) )

        # 2 Initialize the hyperparameters on external values
        self._hyperparam_tuple = HyperParamTuple( p_set=self._hyperparam_space )
        self._hyperparam_tuple.set_values( [p_hp_layers, p_hp_neurons_per_layer] )


## -------------------------------------------------------------------------------------------------
    def _adapt(self, p_dataset : list) -> bool:
        """
        This custom method is intended for explicit adaptation. Here you can specify concrete 
        adaptation data parameters and implement a suitable algorithm. 

        Parameters
        ----------
        p_dataset : list
            A dataset

        Returns
        -------
        adapted : bool
            True, if something has been adapted. False otherwise.
        """

        self.log(Log.C_LOG_TYPE_I, 'Incoming dataset:\n', p_dataset)
        self.log(Log.C_LOG_TYPE_I, 'My current hyperparameters are', self.get_hyperparam().get_values() )
        self.log(Log.C_LOG_TYPE_W, 'By the way: I do not really adapt something;)')
        return False


## -------------------------------------------------------------------------------------------------
    def _adapt_on_event(self, p_event_id: str, p_event_object: Event) -> bool:
        """
        This custom method can be used to adapt something based on an event. The methods needs to be
        registered as an event handler on a separate object that in turn raises the event. See class
        mlpro.bf.events.EventHandler for further information. Context informations can be extracted
        from the related event object in parameter p_event_object.

        Returns
        -------
        adapted : bool
            True, if something has been adapted. False otherwise.
        """

        self.log(Log.C_LOG_TYPE_I, 'Custom adaptation based on event', p_event_id)
        self.log(Log.C_LOG_TYPE_W, 'By the way: I do not really adapt something;)')
        return False


## -------------------------------------------------------------------------------------------------
    def _run(self, p_input : float):
        """
        This custom method implements the operational activites of your model. Here you can specify
        concrete runtime parameters to be processed.
        """

        self.log(Log.C_LOG_TYPE_I, 'Incoming data to be processed:', str(p_input))
        self.log(Log.C_LOG_TYPE_W, 'By the way: I do not really process something;)')


## -------------------------------------------------------------------------------------------------
    def get_accuracy(self) -> float:
        """
        This custom method returns the current accuracy of your model...

        Returns
        -------
        accuracy : float
            Accuracy of the model as a scalar value in interval [0,1]
        """

        return 0.0


## -------------------------------------------------------------------------------------------------
    def _init_plot_2d(self, p_figure: Figure, p_settings: PlotSettings):
        """
        This custom method is intended to prepare a 2-dimensional plot. See classes PlotSettings and
        Plottable of sub-package mlpro.bf.plot or the online documentation for further information.
        """

        self.log(Log.C_LOG_TYPE_W, 'My specific plot initialization...')
        return super()._init_plot_2d(p_figure, p_settings)


## -------------------------------------------------------------------------------------------------
    def _init_plot_3d(self, p_figure: Figure, p_settings: PlotSettings):
        """
        This custom method is intended to prepare a 3-dimensional plot. See classes PlotSettings and
        Plottable of sub-package mlpro.bf.plot or the online documentation for further information.
        """

        self.log(Log.C_LOG_TYPE_W, 'My specific plot initialization...')
        return super()._init_plot_3d(p_figure, p_settings)


## -------------------------------------------------------------------------------------------------
    def _init_plot_nd(self, p_figure: Figure, p_settings: PlotSettings):
        """
        This custom method is intended to prepare a n-dimensional plot. See classes PlotSettings and
        Plottable of sub-package mlpro.bf.plot or the online documentation for further information.
        """

        self.log(Log.C_LOG_TYPE_W, 'My specific plot initialization...')
        return super()._init_plot_nd(p_figure, p_settings)


## -------------------------------------------------------------------------------------------------
    def _update_plot_2d(self, p_settings: PlotSettings, **p_kwargs):
        """
        This custom method is intended to update your 2-dimensional plot. See classes PlotSettings and
        Plottable of sub-package mlpro.bf.plot or the online documentation for further information.
        """

        self.log(Log.C_LOG_TYPE_W, 'My specific plot updates...')
        return super()._update_plot_2d(p_settings, **p_kwargs)


## -------------------------------------------------------------------------------------------------
    def _update_plot_3d(self, p_settings: PlotSettings, **p_kwargs):
        """
        This custom method is intended to update your 3-dimensional plot. See classes PlotSettings and
        Plottable of sub-package mlpro.bf.plot or the online documentation for further information.
        """

        self.log(Log.C_LOG_TYPE_W, 'My specific plot updates...')
        return super()._update_plot_3d(p_settings, **p_kwargs)


## -------------------------------------------------------------------------------------------------
    def _update_plot_nd(self, p_settings: PlotSettings, **p_kwargs):
        """
        This custom method is intended to update your n-dimensional plot. See classes PlotSettings and
        Plottable of sub-package mlpro.bf.plot or the online documentation for further information.
        """

        self.log(Log.C_LOG_TYPE_W, 'My specific plot updates...')
        return super()._update_plot_nd(p_settings, **p_kwargs)




# 1 Preparation of demo/unit test mode
if __name__ == '__main__':
    # 1.1 Parameters for demo mode
    visualize   = True
    logging     = Log.C_LOG_ALL
  
else:
    # 1.2 Parameters for internal unit test
    visualize   = False
    logging     = Log.C_LOG_NOTHING


# 2 Instantiation of your custom model (multitasking is disabled)
mymodel = MyModel( p_range_max = Range.C_RANGE_NONE,
                   p_visualize = visualize, 
                   p_logging = logging,
                   p_hp_layers = 5,
                   p_hp_neurons_per_layer = 10 )


# 3 Pseudo-adaptation based on a dataset
ds = [ (1,2), (3,4), (5,6), (6,7) ]
mymodel.adapt( p_dataset = ds )


# 4 Now we let your model do it's job
mymodel.run( p_input = 42 )

if __name__ == '__main__':
    input( '\nPress ENTER to proceed\n')

