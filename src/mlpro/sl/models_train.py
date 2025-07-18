## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - The integrative middleware framework for standardized machine learning
## -- Package : mlpro.sl
## -- Module  : models_train.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2023-06-13  0.0.0     LSB      Creation
## -- 2023-07-15  1.0.0     LSB      Release
## -- 2025-07-18  1.1.0     DA       Refactoring
## -------------------------------------------------------------------------------------------------

"""
Ver. 1.1.0 (2025-07-18)

This module provides training classes for supervised learning tasks.
"""

import os
from datetime import datetime, timedelta

import matplotlib.pyplot as plt

from mlpro.bf import Log, Mode, ImplementationError
from mlpro.bf.data import DataStoring
from mlpro.bf.plot import DataPlotting
from mlpro.bf.datasets import Dataset
from mlpro.bf.math import BatchElement
from mlpro.bf.ml import Model, Scenario, Training, TrainingResults

from mlpro.sl import SLAdaptiveFunction



# Export list for public API
__all__ = [ 'SLDataStoring',
            'SLDataPlotting',
            'SLScenario',   
            'SLTrainingResults',
            'SLTraining' ]




## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class SLDataStoring(DataStoring):
    """
    Custom data storing class for Supervised Learning.

    Parameters
    ----------
    p_variables:
        List of variables for the data storing.
    """

    C_VAR0 = "Epoch ID"
    C_VAR_CYCLE = "Cycle ID"


## -------------------------------------------------------------------------------------------------
    def __init__(self, p_variables:list):

        self.space = p_variables

        self.variables = [self.C_VAR_CYCLE]

        self.var_space = []
        for dim in self.space:
            self.var_space.append(dim)

        self.variables.extend(self.var_space)
        self.current_epoch = 0

        DataStoring.__init__(self,p_variables=self.variables)


## -------------------------------------------------------------------------------------------------
    def memorize_row(self, p_cycle_id, p_data):

        """
        Memorize a row in the Data Storing Object.

        Parameters
        ----------
        p_cycle_id:int
            Cycle Id.
        p_data:
            Data to be stored.

        """

        self.memorize(self.C_VAR_CYCLE, self.current_epoch, p_cycle_id)


        for i, var in enumerate(self.var_space):
            self.memorize(var, self.current_epoch, p_data[i])


## -------------------------------------------------------------------------------------------------
    def get_variables(self):
        """
        Get the variables for this data storing object.

        Returns
        -------
        variables: list
            List of variables for the data storing object.
        """

        return self.variables


## -------------------------------------------------------------------------------------------------
    def add_epoch(self, p_epoch_id):

        """
        Add epoch to the data storing object. Adds a frame with a new epoch ID.

        Parameters
        ----------
        p_epoch_id: int
            Epoch id for which a new frame is to be added.

        """
        self.add_frame(p_frame_id=p_epoch_id)
        self.current_epoch = p_epoch_id






## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class SLDataPlotting(DataPlotting):

    C_PLOT_TYPE_MULTI_VARIABLE = 'Multi Var'
    C_PLOT_STYLE_SCATTER = 'Scatter'
    C_PLOT_STYLE_LINE = 'Line'


## -------------------------------------------------------------------------------------------------
    def __init__(self,
                 p_data: DataStoring,
                 p_type=DataPlotting.C_PLOT_TYPE_EP,
                 p_window=100,
                 p_showing=True,
                 p_printing=None,
                 p_figsize=(7, 7),
                 p_color="darkblue",
                 p_window_type='same',
                 p_names = [None],
                 p_style = C_PLOT_STYLE_LINE):

        DataPlotting.__init__(self,
                              p_data = p_data,
                              p_type=p_type,
                              p_window=p_window,
                              p_showing=p_showing,
                              p_printing=p_printing,
                              p_figsize=p_figsize,
                              p_color=p_color,
                              p_window_type=p_window_type)

        self._names = p_names
        self._style = p_style

## -------------------------------------------------------------------------------------------------
    def get_plots(self):
        """
        A function to plot data.
        """

        if self.type == 'Cyclic':
            self.plots_type_cy()
        elif self.type == 'Episodic':
            self.plots_type_ep()
        elif self.type == 'Episodic Mean':
            self.plots_type_ep_mean()
        elif self.type == 'Multi Var':
            self.plots_type_multi_variable()


## -------------------------------------------------------------------------------------------------
    def plots_type_multi_variable(self):
        """
        A function to plot data per frame by extending the cyclic plots in one plot.
        """

        labels = []
        fig = plt.figure(figsize=self.figsize)
        for var in self.printing.keys():
            data = []
            plt.title(self._names[0])
            plt.grid(True, which='both', axis = 'both')
            for fr in range(len(self.data.memory_dict[var])):
                fr_id = self.data.frame_id[var][fr]
                data.extend(self.data.get_values(var, fr_id))
            labels.append(var)
            indexes = list(range(1, len(data)+1))
            while True:
                if None in data:
                    index = data.index(None)
                    del data[index]
                    del indexes[index]
                    continue
                else:
                    break
            plt.plot(indexes, data)
        plt.legend(labels, bbox_to_anchor=(1, 0.5), loc="center left")
        self.plots[0].append(self._names[0])
        self.plots[1].append(fig)


        plt.show()






## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class SLScenario (Scenario):
    """
    This is custom Scenario Class specialised for Supervised Learning.

    Parameters
    ----------
    p_mode
        Mode of the Scenario
    p_ada: bool
        Whether the scenario is adaptive or not. Default is True.
    p_cycle_limit: int
        The number of cycles for which the scenario needs to be run.
    p_collect_mappings:bool
        Whether the mappings from the scenario run shall be collected.
    p_collect_cycles:bool
        Whether the model scores foreach cycle shall be collected.
    p_path:str
        Path to which the scenario shall be saved.
    p_get_mapping_plots: bool
        Whether the mapping plots shall be generated and saved.
    p_get_metric_plots:bool
        Whether the metric plots shall be generated and saved.
    p_save_plots:bool
        Whether the plots shall be saved or not.
    p_visualize: bool
        Switch for visualization.
    p_logging
        Log level for the scenario.

    """

    C_TYPE = 'SL-Scenario'
    C_NAME = 'Unnamed'

    C_FNAME_MAPPING = 'Mappings'
    C_FNAME_CYCLES = 'Cycles'


## -------------------------------------------------------------------------------------------------
    def __init__(self,
                 p_mode=Mode.C_MODE_SIM,
                 p_ada: bool = True,
                 p_cycle_limit: int = 0,
                 p_collect_mappings = False,
                 p_collect_cycles = False,
                 p_path = None,
                 p_get_mapping_plots = False,
                 p_get_metric_plots = False,
                 p_save_plots = False,
                 p_visualize: bool = True,
                 p_logging=Log.C_LOG_ALL):

        self._dataset : Dataset = None
        self._model : SLAdaptiveFunction = None
        self._collect_mapping = p_collect_mappings
        self._collect_cycles = p_collect_cycles
        self._mapping_plots = p_get_mapping_plots
        self._metric_plots = p_get_metric_plots
        self.data_plotters = []
        self.save_plots = p_save_plots

        Scenario.__init__(self,
                          p_mode = p_mode,
                          p_ada = p_ada,
                          p_cycle_limit = p_cycle_limit,
                          p_visualize = p_visualize,
                          p_logging = p_logging)

        # TODO: Check if i need a cycle limit specific to models

        self.connect_datalogger()
        # raise NotImplementedError

        if self._dataset is None:
            raise ImplementationError("Please bind your SL dataset to the _dataset attribute in the _setup method.")


        self._metrics = self._model.get_metrics()

        if self._collect_cycles or self._collect_mapping:
            self._setup_datalogging()
            self._path = self._gen_root_path(p_path=p_path)


## -------------------------------------------------------------------------------------------------
    def _run_cycle(self):
        """
        Runs a single cycle of the Supervised Learning Scenario.
        """

        # Check if the first run
        success = False
        error = False

        data = self._dataset.get_next()
        adapted = self._model.adapt(p_dataset = data)

        if self.get_dataset()._last_batch:
            self.log(Log.C_LOG_TYPE_I, "End of Data reached.")
            end_of_data = True

        else:

            end_of_data = False

            self.log(Log.C_LOG_TYPE_I, "Computing the metrics")

            for input, target in data:

                output = self._model(input)
                logging_data = self._model.get_logging_data()

                metric_values = self._model.calculate_metrics(p_data = (input, target)).get_values()

                for met_val in metric_values:
                    logging_data.append(met_val.get_values())

                self.log(Log.C_LOG_TYPE_I, 'Current Metrics',*['\t'+self._metrics[i].get_name() +":\t"+ str(metric_values[i].get_values()) for i in range(len(self._metrics))])

                if self.ds_cycles is not None:
                    self.ds_cycles.memorize_row(p_cycle_id=self.get_cycle_id(), p_data = logging_data)

                if self.ds_mappings is not None:
                    if isinstance(output, BatchElement):
                        for i,val in enumerate(output.get_values()):
                            ip = input.get_values()[i]
                            tg = target.get_values()[i]
                            op = val
                            self.ds_mappings.memorize_row(p_cycle_id=self.get_cycle_id(), p_data= [*ip, *tg, *op])
                    else:
                        self.ds_mappings.memorize_row(p_cycle_id=self.get_cycle_id(), p_data=[*input.get_values(),
                                                                                              *target.get_values(),
                                                                                              *output.get_values()])


        # get success from the model
        # Future implementation to terminate a training based on a goal criterion

        # Error computations such as Stagnation, etc.
        # Future implementation to terminate a training based on an undesirable criterion

        # To be shifted in lower levels of scenario classes
        if end_of_data or (self.get_cycle_id()>=(self._cycle_limit-1)):

            for dp in self.data_plotters:
                dp.get_plots()
                if self.save_plots:
                    dp.save_plots(p_path=self._path, p_format='jpg')

            if self._collect_cycles or self._collect_mapping:
                self.save(p_path=self._path)

        return success, error, adapted, end_of_data


## -------------------------------------------------------------------------------------------------
    def _setup(self, p_mode, p_ada:bool, p_visualize:bool, p_logging) -> Model:
        """
        Setup the scenario in this method. Please rewrite this method to assign the dataset to self._dataset
        attribute and return a model.

        Parameters
        ----------
        p_mode:
            Mode for the simulation.
        p_ada: bool
            Adaptivity switch.
        p_visualize: bool
            Visualization switch.
        p_logging:
            Log level to be set.

        """
        raise NotImplementedError


## -------------------------------------------------------------------------------------------------
    def _reset(self, p_seed):
        """
        Custom reset method, to reset the SLScenario. Resets the Metrics and re-initializes the
        visualization if True.

        Parameters
        ----------
        p_seed: int
            Seed for the purpose of reproducibility.

        """
        if self._visualize:
            self._model.init_plot()

        for metric in self._metrics:
            metric.reset(p_seed)


## -------------------------------------------------------------------------------------------------
    def connect_datalogger(self, p_mapping:SLDataStoring = None, p_cycle:SLDataStoring = None):

        """
        Connect the datastoring objects to the scenario.

        Parameters
        ----------
        p_mapping:
            Datastoring object for collecting Mappings.

        p_cycle:
            Datastoring object for collecting model scores for each cycle.

        """

        self.ds_mappings = p_mapping
        self.ds_cycles = p_cycle


## -------------------------------------------------------------------------------------------------
    def get_dataset(self):

        """
        Get the dataset assigned to the scenario.

        Returns
        -------
        Dataset:
            Dataset object assigned to the scenario.
        """

        return self._dataset


## -------------------------------------------------------------------------------------------------
    def _init_plot(self):

        """
        Initializes the plot.
        """

        pass


## -------------------------------------------------------------------------------------------------
    def _update_plot(self):

        """
        Updates the plot.
        """

        pass


## -------------------------------------------------------------------------------------------------
    def get_latency(self):

        """
        Gets the latency of the Scenario.
        """

        return timedelta(0,0,0)


## -------------------------------------------------------------------------------------------------
    def _gen_root_path(self, p_path = None):
        """
        Generates the root path, in case data is to be saved.

        Parameters
        ----------
        p_path: str
            destination Path for saving the data

        Returns
        -------
        root_path:str
            The root path generated.
        """

        if p_path is None: return None

        now = datetime.now()
        ts = '%04d-%02d-%02d  %02d.%02d.%02d' % (now.year, now.month, now.day, now.hour, now.minute, now.second)
        root_path = p_path + str(os.sep) + ts + ' ' + self.C_TYPE + ' ' + self.C_NAME
        root_path.replace(str(os.sep) + str(os.sep), str(os.sep))
        os.mkdir(root_path)
        return root_path


## -------------------------------------------------------------------------------------------------
    def _reduce_state(self, p_state:dict, p_path:str, p_os_sep:str, p_filename_stub:str):
        """
        Reduces the state of the object, before pickling, to avoid saving complex/incompatible/unnecessary objects.

        Parameters
        ----------
        p_state:dict
            State dict of the object.
        p_path:str
            Path where the object is to be saved.
        p_os_sep:str
            The path separator for the particular Operating system.
        p_filename_stub:str
            Filename stub (filename without extension) for the file

        """
        Scenario._reduce_state(self, p_state=p_state, p_path=p_path, p_filename_stub=p_filename_stub, p_os_sep=p_os_sep)

        p_state['data_plotters'] = None

        if self._collect_mapping:
            p_state['ds_mappings'].save_data(p_path=p_path, p_filename=self.C_FNAME_CYCLES, p_delimiter='\t')
            p_state['ds_mappings'] = None

        if self._collect_cycles:
            p_state['ds_cycles'].save_data(p_path=p_path, p_filename=self.C_FNAME_MAPPING, p_delimiter='\t')
            p_state['ds_cycles'] = None


## -------------------------------------------------------------------------------------------------
    def _setup_datalogging(self):
        """
        Setup the data storing objects for SLScenario, in case data is to be collected.

        """

        self._save = True
        variables = []

        # 1. Creating data storing for Mappings
        if self._collect_mapping:
            for dim in self._model.get_input_space().get_dims():
                variables.append('input '+dim.get_name_short())
            for dim in self._model.get_output_space().get_dims():
                variables.append('target '+dim.get_name_short())
            for dim in self._model.get_output_space().get_dims():
                variables.append('pred '+dim.get_name_short())
            self.ds_mappings = SLDataStoring(p_variables=variables)
            self.ds_mappings.add_frame(p_frame_id=0)
        if self._mapping_plots:
            for dim in self._model.get_output_space().get_dims():
                self.data_plotters.append(SLDataPlotting(p_data=self.ds_mappings,
                                                         p_type=SLDataPlotting.C_PLOT_TYPE_MULTI_VARIABLE,
                                                         p_printing={'target '+dim.get_name_short():[True, 0, -1],
                                                                     'pred '+dim.get_name_short() : [True, 0, -1]},
                                                         p_names=[dim.get_name_short()]))
        logging_variables = []
        # 2. Creating data storing for Cycle Scores
        if self._collect_cycles:
            for dim in self._model.get_logging_set().get_dims():
                logging_variables.append(dim.get_name_short())
            for dim in self._model.get_metric_space().get_dims():
                logging_variables.append(dim.get_name_short())
            self.ds_cycles = SLDataStoring(p_variables=logging_variables)
            self.ds_cycles.add_frame(p_frame_id=0)

        # 3. Creating data plotting objects
        if self._metric_plots:
            printing = {}
            for logging_variable in logging_variables:
                printing[logging_variable] = [True, 0, -1]
            self.data_plotters.append(SLDataPlotting(p_data=self.ds_cycles,
                                                     p_type=SLDataPlotting.C_PLOT_TYPE_EP,
                                                     p_printing=printing,
                                                     p_names=logging_variables))






## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class SLTrainingResults(TrainingResults):
    """
    The custom training results object specialised for Supervised Learning Training.

    Parameters
    ----------
    p_scenario: Scnenario
        The scenario for the training.
    p_run:
        The training run, in case of Hyperparameter Tuning.
    p_cycle_id:
        The cycle id of the training.
    p_logging:
        Log level for the Results.
    """

    C_NAME = "SL"

    C_FNAME_EPOCH = 'Epoch Scores'
    C_FNAME_TRAIN_SCORE = 'Training Scores'
    C_FNAME_EVAL_SCORE = 'Evaluation Scores'
    C_FNAME_TEST_SCORE = 'Test Scores'
    C_FNAME_TRAIN_MAP = 'Training Predictions'
    C_FNAME_EVAL_MAP = 'Evaluation Predictions'
    C_FNAME_TEST_MAP = 'Test Predictions'

    C_CPAR_NUM_EPOCH_TRAIN = 'Training Epochs'
    C_CPAR_NUM_EPOCH_EVAL = 'Evaluation Epochs'
    C_CPAR_NUM_EPOCH_TEST = 'Test Epochs'


## -------------------------------------------------------------------------------------------------
    def __init__(self,
                 p_scenario: SLScenario,
                 p_run,
                 p_cycle_id,
                 p_logging = Log.C_LOG_WE ):


        TrainingResults.__init__(self,
                                 p_scenario=p_scenario,
                                 p_run=p_run,
                                 p_cycle_id=p_cycle_id,
                                 p_logging=p_logging)

        self.num_epochs_train = 0
        self.num_epochs_eval = 0
        self.num_epochs_test = 0
        self.ds_epoch = None
        self.ds_cycles_train = None
        self.ds_cycles_eval = None
        self.ds_cycles_test = None
        self.ds_mapping_train = None
        self.ds_mapping_eval = None
        self.ds_mapping_test = None
        self.data_plotters = []


## -------------------------------------------------------------------------------------------------
    def close(self):
        """
        Close the results object by logging the results in the console.
        """

        TrainingResults.close(self)

        self.add_custom_result(self.C_CPAR_NUM_EPOCH_TRAIN, self.num_epochs_train)
        self.add_custom_result(self.C_CPAR_NUM_EPOCH_EVAL, self.num_epochs_eval)
        self.add_custom_result(self.C_CPAR_NUM_EPOCH_TEST, self.num_epochs_test)


## -------------------------------------------------------------------------------------------------
    def _log_results(self):
        """
        Custom method to log the training results, with additional results to be logged.
        """
        TrainingResults._log_results(self)

        self.log(Log.C_LOG_WE, "Training Epochs:", self.num_epochs_train)
        self.log(Log.C_LOG_WE, "Evaluation Epochs:", self.num_epochs_eval)
        self.log(Log.C_LOG_WE, "Test Epochs:", self.num_epochs_test)


## -------------------------------------------------------------------------------------------------
    def save(self, p_path, p_filename = 'summary.csv') -> bool:

        """
        Save the training results.

        Parameters
        ----------
        p_path:str
            Path where the results shall be saved.
        p_filename:str
            Filename to store the results.

        Returns
        -------
        bool
            True if saved successfully.
        """

        if not TrainingResults.save(self, p_path = p_path, p_filename = p_filename):
            return False

        if self.ds_epoch is not None:
            self.ds_epoch.save_data(p_path, self.C_FNAME_EPOCH)
        if self.ds_cycles_train is not None:
            self.ds_cycles_train.save_data(p_path, self.C_FNAME_TRAIN_SCORE)
        if self.ds_cycles_eval is not None:
            self.ds_cycles_eval.save_data(p_path, self.C_FNAME_EVAL_SCORE)
        if self.ds_cycles_test is not None:
            self.ds_cycles_test.save_data(p_path, self.C_FNAME_TEST_SCORE)
        if self.ds_mapping_train is not None:
            self.ds_mapping_train.save_data(p_path, self.C_FNAME_TRAIN_MAP)
        if self.ds_mapping_eval is not None:
            self.ds_mapping_eval.save_data(p_path, self.C_FNAME_EVAL_MAP)
        if self.ds_mapping_test is not None:
            self.ds_mapping_test.save_data(p_path, self.C_FNAME_TEST_MAP)
        for dp in self.data_plotters:
            dp.get_plots()
            dp.save_plots(p_path = p_path, p_format = 'jpg')






## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class SLTraining (Training):

    """
    Custom training class specialised for Supervised Learning.

    Parameters
    ----------
    p_collect_epoch_scores:bool
        Whether epoch scores shall be collected.
    p_collect_mappings:bool
        Whether mappings shall be collected.
    p_collect_cycles:bool
        Whether cycle scores shall be collected.
    p_plot_epoch_scores:
        Whether epoch scores shall be plotted.
    p_plot_mappings:
        Whether mappings shall be plotted.
    p_plot_cycles:
        Whether cycle score shall be plotted.
    p_maximize_score:
        Which schore is to be maximized. Valid values are:
        C_TRAIN_SCORE: Training Score
        C_EVAL_SCORE: Evaluation Score
        C_TEST_SCORE: Test Score
    p_num_epoch:
        Number of epochs.
    p_eval_freq:
        Evaluatio  frequency.
    p_test_freq:
        Test frequency.
    p_kwargs:
        Additional Training Parameters.
    """

    C_NAME = 'SL'
    C_MODE_TEST = 2

    C_CLS_RESULTS = SLTrainingResults

    C_TEST_SCORE = 'Test'
    C_TRAIN_SCORE = 'Train'
    C_EVAL_SCORE = 'Evaluation'

## -------------------------------------------------------------------------------------------------
    def __init__(self,
                 p_collect_epoch_scores = True,
                 p_collect_mappings = False,
                 p_collect_cycles = False,
                 p_plot_epoch_scores = False,
                 p_plot_mappings = False,
                 p_plot_cycles = False,
                 p_maximize_score = C_EVAL_SCORE,
                 p_num_epoch = 1,
                 p_eval_freq = 0,
                 p_test_freq = 0,
                 **p_kwargs):

        self._collect_epoch_scores = p_collect_epoch_scores
        self._collect_mappings = p_collect_mappings
        self._collect_cycles = p_collect_cycles

        self._plot_epoch_score = p_plot_epoch_scores
        self._plot_mappings = p_plot_mappings
        self._plot_cycle = p_plot_cycles

        self._eval_freq = p_eval_freq
        self._test_freq = p_test_freq

        self._num_epochs = p_num_epoch
        self._cycles_epoch = 0

        self._scenario : SLScenario = None
        self._model : SLAdaptiveFunction = None

        Training.__init__(self, **p_kwargs)

        # For hpt
        if self._hpt is None:
            self._model: SLAdaptiveFunction = self.get_scenario()._model
            self.metric_space = self._model.get_metric_space()
            self._score_metric = self._model.get_score_metric()
            self._logging_space = self._model.get_logging_set()


        self._eval_freq = p_eval_freq
        self._test_freq = p_test_freq
        self._epoch_id = 0
        self._epoch_train = False
        self._epoch_test = False
        self._epoch_eval = False
        self._maximize_score = p_maximize_score


## -------------------------------------------------------------------------------------------------
    def _init_results(self) -> TrainingResults:
        """
        Initialize the training results object, in this case SLTrainingResults.

        Returns
        -------
        results: TrainingResults
            The SLTraining Results object created.
        """

        results = Training._init_results(self)

        results.num_cycles_test = 0

        results._ds_list = []
        metric_variables = [i.get_name_short() for i in self.metric_space.get_dims()]
        # 1. Creating data storing objects for Epoch scores
        if self._collect_epoch_scores:
            variables = metric_variables[:]
            if self._eval_freq > 0:
                for i in range(len(variables)):
                    variables.append("Eval "+ variables[i])
            if self._test_freq > 0:
                for i in range(len(variables)):
                    variables.append("Test " + variables[i])
            results.ds_epoch = SLDataStoring(variables)
            results._ds_list.append(results.ds_epoch)

            # 1.1 Creating data plotting objects for Epoch Scores
            if self._plot_epoch_score:
                if (not self._eval_freq) and (not self._test_freq):
                    printing = {}
                    for var in variables:
                        printing[var] = [True, 0, -1]
                    results.data_plotters.append(SLDataPlotting(p_data=results.ds_epoch,
                                                                p_type=SLDataPlotting.C_PLOT_TYPE_EP,
                                                                p_printing=printing,
                                                                p_names=list(printing.keys())))
                else:
                    for var in metric_variables:
                        printing = {var : [True, 0 -1]}
                        if ("Eval "+var) in variables:
                            printing["Eval "+var] = [True, 0, -1]
                        if ("Test "+var) in variables:
                            printing["Test "+var] = [True, 0, -1]

                        results.data_plotters.append(SLDataPlotting(p_data=results.ds_epoch,
                                                                    p_type=SLDataPlotting.C_PLOT_TYPE_MULTI_VARIABLE,
                                                                    p_printing=printing,
                                                                    p_names=[var]))

        # 2. Creating data storing objects for cycle data
        if self._collect_cycles:
            variables = [i.get_name_short() for i in self._logging_space.get_dims()]
            variables.extend(metric_variables)
            results.ds_cycles_train = SLDataStoring(p_variables=variables)
            results._ds_list.append(results.ds_cycles_train)

            if self._eval_freq > 0:
                results.ds_cycles_eval = SLDataStoring(p_variables=variables)
                results._ds_list.append(results.ds_cycles_eval)

            if self._test_freq > 0:
                results.ds_cycles_test = SLDataStoring(p_variables=variables)
                results._ds_list.append(results.ds_cycles_test)

        # 3. Creating data storing objects for mappings
        if self._collect_mappings:
            variables = []
            for dim in self._model.get_input_space().get_dims():
                variables.append("input "+dim.get_name_short())
            for dim in self._model.get_output_space().get_dims():
                variables.append("target "+dim.get_name_short())
            for dim in self._model.get_output_space().get_dims():
                variables.append("pred "+dim.get_name_short())

            results.ds_mapping_train = SLDataStoring(p_variables=variables)
            results._ds_list.append(results.ds_mapping_train)
            if self._eval_freq > 0:
                results.ds_mapping_eval = SLDataStoring(p_variables=variables)
                results._ds_list.append(results.ds_mapping_eval)
            if self._test_freq > 0:
                results.ds_mapping_test = SLDataStoring(p_variables=variables)
                results._ds_list.append(results.ds_mapping_test)

        # 4. Connect data loggers
        self._scenario.connect_datalogger(p_mapping = results.ds_mapping_train, p_cycle = results.ds_cycles_train)


        return results


## -------------------------------------------------------------------------------------------------
    def _run_cycle(self) -> bool:

        """
        Run one training cycle.

        Returns
        -------
        bool
            True if the training is finished.
        """

        eof_training = False
        eof_epoch = False

        # 1. New epoch
        if self._cycles_epoch == 0:
            self._init_epoch()

        # 2. Run one SLScenario cycle
        success, error, timeout, limit, adapted, end_of_data = self._scenario.run_cycle()
        self._cycles_epoch += 1

        if self._mode == self.C_MODE_TRAIN:
            self._counter_train_cycles += 1

        elif self._mode == self.C_MODE_EVAL:
            self._counter_eval_cycles += 1

        if self._mode == self.C_MODE_TEST:
            self._results.num_cycles_test += 1
            self._counter_test_cycles += 1


        if adapted:
            self._results.num_adaptations += 1


        if end_of_data:
            self._update_scores()
            self.get_scenario().reset(p_seed=self._epoch_id)


            if self._mode == self.C_MODE_TRAIN:
                self._results.num_epochs_train += 1
                for i, dim in enumerate(self.metric_space.get_dims()):
                    self.log(Log.C_LOG_WE, dim.get_name_short(), ':\t', self._train_epoch_scores[i])

                self.log(self.C_LOG_TYPE_W, Training.C_LOG_SEPARATOR)
                self.log(self.C_LOG_TYPE_W, '-- Training epoch', self._epoch_id, 'finished after',
                             str(self._counter_train_cycles), 'cycles')
                self.log(self.C_LOG_TYPE_W, '-- Training cycles finished:', self._results.num_cycles_train + 1)
                self.log(self.C_LOG_TYPE_W, Training.C_LOG_SEPARATOR, '\n\n')



            elif self._mode == self.C_MODE_EVAL:
                for i, dim in enumerate(self.metric_space.get_dims()):
                    self.log(Log.C_LOG_WE, dim.get_name_short(), ':\t', self._eval_epoch_scores[i])
                self._results.num_epochs_eval += 1
                self._epoch_eval = False


                self.log(self.C_LOG_TYPE_W, Training.C_LOG_SEPARATOR)
                self.log(self.C_LOG_TYPE_W, '-- Evaluation epoch', self._epoch_id, 'finished after',
                     str(self._counter_eval_cycles), 'cycles')
                self.log(self.C_LOG_TYPE_W, Training.C_LOG_SEPARATOR, '\n\n')



            elif self._mode == self.C_MODE_TEST:
                for i, dim in enumerate(self.metric_space.get_dims()):
                    self.log(Log.C_LOG_WE, dim.get_name_short(),':\t' , self._test_epoch_scores[i])
                self._results.num_epochs_test += 1
                self._epoch_test = False


                self.log(self.C_LOG_TYPE_W, Training.C_LOG_SEPARATOR)
                self.log(self.C_LOG_TYPE_W, '-- Test epoch', self._epoch_id, 'finished after',
                         str(self._counter_test_cycles), 'cycles')
                self.log(self.C_LOG_TYPE_W, Training.C_LOG_SEPARATOR, '\n\n')


            if self._epoch_eval:
                self._mode = self.C_MODE_EVAL
                self._init_eval()


            elif self._epoch_test:
                self._mode = self.C_MODE_TEST
                self._init_test()


            else:
                eof_epoch = True


        if eof_epoch:
            self._close_epoch()

        if (self._adaptation_limit > 0) and (self._results.num_adaptations == self._adaptation_limit):
            self.log(self.C_LOG_TYPE_W, 'Adaptation limit ', str(self._adaptation_limit), ' reached')
            eof_training = True

        elif eof_epoch and self._epoch_id > 0 and self._epoch_id == self._num_epochs:
            self.log(self.C_LOG_TYPE_W, 'Epoch limit ', str(self._num_epochs), ' reached')
            eof_training = True

        return eof_training


## -------------------------------------------------------------------------------------------------
    def _init_epoch(self):

        """
        Initializes the epoch.
        """

        self._counter_train_cycles = 0
        self._counter_eval_cycles = 0
        self._counter_test_cycles = 0

        if self._epoch_id == 0:
            self.log(self.C_LOG_TYPE_W, Training.C_LOG_SEPARATOR)
            self.log(self.C_LOG_TYPE_W, '-- Training period started...')
            self.log(self.C_LOG_TYPE_W, Training.C_LOG_SEPARATOR, '\n')

        self._epoch_id += 1
        self._mode = self.C_MODE_TRAIN

        self.log(self.C_LOG_TYPE_W, Training.C_LOG_SEPARATOR)
        self.log(self.C_LOG_TYPE_W, '-- Training epoch', self._epoch_id, 'started...')
        self.log(self.C_LOG_TYPE_W, Training.C_LOG_SEPARATOR, '\n')

        self._model.switch_adaptivity(p_ada = True)

        self._scenario.get_dataset().reset(p_seed=self._epoch_id)

        # Check if this is also an evaluation epoch
        if self._eval_freq:
            if self._epoch_id % self._eval_freq == 0:
                self._epoch_eval = True

        # Check if this is also a training epoch
        if self._test_freq:
            if self._epoch_id % self._test_freq == 0:
                self._epoch_test = True

        for ds in self._results._ds_list:
            ds.add_epoch(self._epoch_id)

        self._scenario.connect_datalogger(p_mapping = self._results.ds_mapping_train,
                                          p_cycle = self._results.ds_cycles_train)


        self._scenario.get_dataset().set_mode(Dataset.C_MODE_TRAIN)

        self._train_epoch_scores = [None for i in range(len(self.metric_space.get_dims()))]
        self._test_epoch_scores = [None for i in range(len(self.metric_space.get_dims()))]
        self._eval_epoch_scores = [None for i in range(len(self.metric_space.get_dims()))]



## -------------------------------------------------------------------------------------------------
    def _update_scores(self):

        """
        Update the scores of the Model during the training.
        """
        current_metrics = []
        for metric in self._model.get_metrics():
            current_metrics.append(metric.get_current_score())
        if self._mode == self.C_MODE_TRAIN:
            self._train_epoch_scores = current_metrics
            self._train_highscore = self._score_metric.get_current_highscore()
        elif self._mode == self.C_MODE_TEST:
            self._test_epoch_scores = current_metrics
            self._test_highscore = self._score_metric.get_current_highscore()
        elif self._mode == self.C_MODE_EVAL:
            self._eval_epoch_scores = current_metrics
            self._eval_highscore = self._score_metric.get_current_highscore()


## -------------------------------------------------------------------------------------------------
    def _close_epoch(self):

        """
        Close the epoch and update the highscore.
        """

        score = [*self._train_epoch_scores, *self._eval_epoch_scores, *self._test_epoch_scores]


        self._results.ds_epoch.memorize_row(self._scenario.get_cycle_id(), p_data=score)

        if self._maximize_score == self.C_TRAIN_SCORE:
            self._results.highscore = self._train_highscore

        elif self._maximize_score == self.C_EVAL_SCORE:
            if self._eval_freq > 0:
                self._results.highscore = self._eval_highscore
            else:
                raise ImplementationError("Evaluation frequency must be greater than zero "
                                          "for Evaluation score maximization")
        elif self._maximize_score == self.C_TEST_SCORE:
            if self._test_freq > 0:
                self._results.highscore = self._eval_highscore
            else:
                raise ImplementationError("Test frequency must be greater than zero "
                                          "for Test score maximization")

        self._cycles_epoch = 0


## -------------------------------------------------------------------------------------------------
    def _init_eval(self):

        """
        Inintialize evaluation epoch.
        """
        self.log(self.C_LOG_TYPE_W, Training.C_LOG_SEPARATOR)
        self.log(self.C_LOG_TYPE_W, '-- Evaluation epoch', self._epoch_id, 'started...')
        self.log(self.C_LOG_TYPE_W, Training.C_LOG_SEPARATOR, '\n')
        self._model.switch_adaptivity(p_ada=False)
        self._mode = self.C_MODE_EVAL
        self._scenario.connect_datalogger(p_mapping=self._results.ds_mapping_eval, p_cycle=self._results.ds_cycles_eval)
        self._scenario.get_dataset().set_mode(Dataset.C_MODE_EVAL)


## -------------------------------------------------------------------------------------------------
    def _init_test(self):

        """
        Initialize the test epoch.
        """
        self.log(self.C_LOG_TYPE_W, Training.C_LOG_SEPARATOR)
        self.log(self.C_LOG_TYPE_W, '-- Test epoch', self._epoch_id, 'started...')
        self.log(self.C_LOG_TYPE_W, Training.C_LOG_SEPARATOR, '\n')
        self._model.switch_adaptivity(p_ada=False)
        self._mode = self.C_MODE_TEST
        self._scenario.connect_datalogger(p_mapping=self._results.ds_mapping_test, p_cycle=self._results.ds_cycles_test)
        self._scenario.get_dataset().set_mode(Dataset.C_MODE_TEST)




