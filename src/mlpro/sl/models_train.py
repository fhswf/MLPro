## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - A Synoptic Framework for Standardized Machine Learning Tasks
## -- Package : mlpro.sl
## -- Module  : basics.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2023-06-13  0.0.0     LSB      Creation
## -------------------------------------------------------------------------------------------------

"""
Ver. 0.0.0 (2023-06-13)

This module provides training classes for supervised learning tasks.
"""
import warnings

import matplotlib.pyplot as plt

from mlpro.bf.data import *
from mlpro.sl import *
from mlpro.bf.plot import DataPlotting





## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class SLDataStoring(DataStoring):
    """

    Parameters
    ----------
    p_variables
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

        Parameters
        ----------
        p_cycle_id
        p_data

        """

        self.memorize(self.C_VAR_CYCLE, self.current_epoch, p_cycle_id)


        for i, var in enumerate(self.var_space):
            self.memorize(var, self.current_epoch, p_data[i])


## -------------------------------------------------------------------------------------------------
    def get_variables(self):
        """

        Returns
        -------

        """
        return self.variables


## -------------------------------------------------------------------------------------------------
    def add_epoch(self, p_epoch_id):

        """

        Parameters
        ----------
        p_epoch_id

        """
        self.add_frame(p_frame_id=p_epoch_id)
        self.current_epoch = p_epoch_id






## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class SLDataPlotting(DataPlotting):
    C_PLOT_TYPE_MULTI_VARIABLE = 'Multi Var'
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


    def plots_type_multi_variable(self):
        """
        A function to plot data per frame by extending the cyclic plots in one plot.
        """

        labels = []
        fig = plt.figure(figsize=self.figsize)
        for var in self.printing.keys():
            data = []
            indexes = []
            plt.title('custom multi-variable plot')
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
        self.plots[0].append('Multi-Variable')
        self.plots[1].append(fig)


        plt.show()






## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class SLScenario (Scenario):
    """
    To be designed.

    Parameters
    ----------
    p_mode
    p_ada
    p_cycle_limit
    p_visualize
    p_logging

    """

    C_TYPE = 'SL-Scenario'


## -------------------------------------------------------------------------------------------------
    def __init__(self,
                 p_mode=Mode.C_MODE_SIM,
                 p_ada: bool = True,
                 p_cycle_limit: int = 0,
                 p_visualize: bool = True,
                 p_logging=Log.C_LOG_ALL):

        self._dataset : Dataset = None
        self._model : SLAdaptiveFunction = None
        # self.ds : dict = {}

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


## -------------------------------------------------------------------------------------------------
    def _run_cycle(self):

        """

        """
        # Check if the first run
        success = False
        error = False

        data = self._dataset.get_next()
        adapted = self._model.adapt(p_dataset = data)

        if self.get_dataset()._last_batch:
            end_of_data = True

        else:

            end_of_data = False

            for input, target in data:

                output = self._model(input)
                logging_data = self._model.get_logging_data()
                # mapping = [*input.get_values(), *target.get_values(), *output.get_values()]


                metric_values = self._model.calculate_metrics(p_data = (input, target)).get_values()

                for met_val in metric_values:
                    logging_data.append(met_val.get_values())

                if self.get_cycle_id() == 0:
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

        return success, error, adapted, end_of_data


## -------------------------------------------------------------------------------------------------
    def _setup(self, p_mode, p_ada:bool, p_visualize:bool, p_logging) -> Model:
        """

        Parameters
        ----------
        p_mode
        p_ada
        p_visualize
        p_logging

        Returns
        -------

        """
        raise NotImplementedError


## -------------------------------------------------------------------------------------------------
    def _reset(self, p_seed):
        """

        Parameters
        ----------
        p_seed

        """
        if self._visualize:
            self._model.init_plot()

        for metric in self._metrics:
            metric.reset(p_seed)


## -------------------------------------------------------------------------------------------------
    def connect_datalogger(self, p_mapping:DataStoring = None, p_cycle:DataStoring = None):

        """

        Parameters
        ----------
        p_mapping
        p_cycle

        """
        self.ds_mappings = p_mapping
        self.ds_cycles = p_cycle


## -------------------------------------------------------------------------------------------------
    def get_dataset(self):

        """

        Returns
        -------

        """
        return self._dataset


## -------------------------------------------------------------------------------------------------
    def _init_plot(self):

        """

        """
        pass


## -------------------------------------------------------------------------------------------------
    def _update_plot(self):

        """

        """
        pass


## -------------------------------------------------------------------------------------------------
    def get_latency(self):

        """


        """
        return timedelta(0,0,0)





## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class SLTrainingResults(TrainingResults):

    """

    Parameters
    ----------
    p_scenario
    p_run
    p_cycle_id
    p_logging
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

## -------------------------------------------------------------------------------------------------
    def close(self):

        """

        """
        TrainingResults.close(self)

        self.add_custom_result(self.C_CPAR_NUM_EPOCH_TRAIN, self.num_epochs_train)
        self.add_custom_result(self.C_CPAR_NUM_EPOCH_EVAL, self.num_epochs_eval)
        self.add_custom_result(self.C_CPAR_NUM_EPOCH_TEST, self.num_epochs_test)


## -------------------------------------------------------------------------------------------------
    def _log_results(self):

        """


        """

        TrainingResults._log_results(self)

        self.log(Log.C_LOG_WE, "Training Epochs:", self.num_epochs_train)
        self.log(Log.C_LOG_WE, "Evaluation Epochs:", self.num_epochs_eval)
        self.log(Log.C_LOG_WE, "Test Epochs:", self.num_epochs_test)



## -------------------------------------------------------------------------------------------------
    def save(self, p_path, p_filename = 'summary.csv') -> bool:

        """

        Parameters
        ----------
        p_path
        p_filename

        Returns
        -------

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







## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class SLTraining (Training):
    """

    Parameters
    ----------
    p_collect_epoch_scores
    p_collect_mappings
    p_collect_cycles
    p_eval_freq
    p_test_freq
    p_kwargs
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
                 p_maximize_score = C_TRAIN_SCORE,
                 p_num_epoch = 1,
                 p_eval_freq = 0,
                 p_test_freq = 0,
                 **p_kwargs):


        self._collect_epoch_scores = p_collect_epoch_scores
        self._collect_mappings = p_collect_mappings
        self._collect_cycles = p_collect_cycles

        self._eval_freq = p_eval_freq
        self._test_freq = p_test_freq

        self._num_epochs = p_num_epoch
        self._cycles_epoch = 0

        self._scenario : SLScenario = None
        self._model : SLAdaptiveFunction = None

        Training.__init__(self, **p_kwargs)

        if self._hpt is None:
            self._model: SLAdaptiveFunction = self.get_scenario()._model

            self.metric_space = self._model.get_metric_space()


        # For hpt
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

        Returns
        -------

        """

        results = Training._init_results(self)

        results.num_cycles_test = 0

        results._ds_list = []
        metric_variables = [i.get_name_short() for i in self.metric_space.get_dims()]
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

        self._scenario.connect_datalogger(p_mapping = results.ds_mapping_train, p_cycle = results.ds_cycles_train)


        return results


## -------------------------------------------------------------------------------------------------
    def _run_cycle(self) -> bool:

        """

        Returns
        -------

        """
        eof_training = False
        eof_epoch = False

        if self._cycles_epoch == 0:
            self._init_epoch()

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

        if self._eval_freq:
            if self._epoch_id % self._eval_freq == 0:
                self._epoch_eval = True

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

        """
        self.log(self.C_LOG_TYPE_W, Training.C_LOG_SEPARATOR)
        self.log(self.C_LOG_TYPE_W, '-- Test epoch', self._epoch_id, 'started...')
        self.log(self.C_LOG_TYPE_W, Training.C_LOG_SEPARATOR, '\n')
        self._model.switch_adaptivity(p_ada=False)
        self._mode = self.C_MODE_TEST
        self._scenario.connect_datalogger(p_mapping=self._results.ds_mapping_test, p_cycle=self._results.ds_cycles_test)
        self._scenario.get_dataset().set_mode(Dataset.C_MODE_TEST)




