## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - A Synoptic Framework for Standardized Machine Learning Tasks
## -- Package : mlpro.sl
## -- Module  : basics.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2023-06-13  0.0.0     LSB       Creation
## -------------------------------------------------------------------------------------------------

"""
Ver. 0.0.0 (2023-06-13)

This module provides training classes for supervised learning tasks.
"""


from mlpro.bf.data import *
from mlpro.sl.basics import *





## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class SLDataStoring(DataStoring):


    C_VAR0 = "Epoch ID"
    C_VAR_CYCLE = "Cycle ID"
    C_VAR_DAY = "Day"
    C_VAR_SEC = "Second"
    C_VAR_MICROSEC = "Microsecond"
    C_VAR_LEARN_RATE = "Learning Rate"
    C_VAR_LOSS = "Loss"

    def __init__(self, p_metric_space):

        self.space = p_metric_space.get_space()

        self.variables = [self.C_VAR_CYCLE, self.C_VAR_DAY, self.C_VAR_SEC, self.C_VAR_MICROSEC,
                          self.C_VAR_LEARN_RATE, self.C_VAR_LEARN_RATE]

        self.var_space = []
        for dim in self.space.get_dims():
            self.var_space.append(dim.get_name_short())

        self.variables.extend(self.var_space)

        DataStoring.__init__(self,p_variables=self.variables)


## -------------------------------------------------------------------------------------------------
    def memorize_row(self, p_cycle_id, p_tstamp, p_lr, p_loss, p_data):


        self.memorize(self.C_VAR_CYCLE, self.current_epoch, p_cycle_id)
        self.memorize(self.C_VAR_DAY, self.current_epoch, p_tstamp.days)
        self.memorize(self.C_VAR_SEC, self.current_epoch, p_tstamp.seconds)
        self.memorize(self.C_VAR_MICROSEC, self.current_epoch, p_tstamp.microseconds)
        self.memorize(self.C_VAR_LEARN_RATE, self.current_epoch, p_lr)
        self.memorize(self.C_VAR_LOSS, self.current_epoch, p_loss)

        for i, var in enumerate(self.var_space):
            self.memorize(var, self.current_epoch, p_data[i])


## -------------------------------------------------------------------------------------------------
    def get_variables(self):
        return self.variables


## -------------------------------------------------------------------------------------------------
    def add_epoch(self, p_epoch_id):
        self.add_frame(p_frame_id=p_epoch_id)
        self.current_epoch = p_epoch_id






## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class SLScenario (Scenario):
    """
    To be designed.
    """

    C_TYPE = 'SL-Scenario'


## -------------------------------------------------------------------------------------------------
    def __init__(self,
                 p_mode=Mode.C_MODE_SIM,
                 p_ada: bool = True,
                 p_cycle_limit: int = 0,
                 p_visualize: bool = True,
                 p_logging=Log.C_LOG_ALL):

        # If dataset config given, setup dataset
        # Assign Datasets
        self._dataset = None
        # save the lengths of all the datasets
        self._model : SLAdaptiveFunction = None
        self._data_loggers : dict = {}
        Scenario.__init__(self,
                          p_mode = p_mode,
                          p_ada = p_ada,
                          p_cycle_limit = p_cycle_limit,
                          p_visualize = p_visualize,
                          p_logging = p_logging)

        self.connect_dataloggers()
        # raise NotImplementedError


## -------------------------------------------------------------------------------------------------
    def _run_cycle(self):

        # Check if the first run
        success = False
        error = False
        adapted = False
        end_of_data = False

        # if self._cycle_id >= ( len(self.get_dataset()) - 1 ):
        #     end_of_data = True
        # else:
        #     end_of_data = False

        # adapted = self._model.adapt(p_dataset = self.get_dataset().get_next())

        # Need to optimize the adapt method of the SLAdaptiveFunction which currently just adapts only when the
        # distance is more than threshold

        # get success from the model

        # Error computations such as Stagnation, etc.

        return success, error, adapted, end_of_data


## -------------------------------------------------------------------------------------------------
    def _setup(self, p_mode, p_ada:bool, p_visualize:bool, p_logging) -> Model:
        raise NotImplementedError


## -------------------------------------------------------------------------------------------------
    def _reset(self, p_seed):

        self._model.reset(p_seed)
        if self._visualize:
            self._model.init_plot()


## -------------------------------------------------------------------------------------------------
    def connect_dataloggers(self, p_ds = None, p_name = None):

        if p_ds is not None and p_name is not None:
            self._data_loggers[p_name] = p_ds


## -------------------------------------------------------------------------------------------------
    def get_dataset(self):
        return self._dataset


## -------------------------------------------------------------------------------------------------
    def _init_plot(self):
        pass


## -------------------------------------------------------------------------------------------------
    def _update_plot(self):
        pass





## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class SLTrainingResults(TrainingResults):
    pass


## -------------------------------------------------------------------------------------------------
    def close(self):
        pass


## -------------------------------------------------------------------------------------------------
    def _log_results(self):
        pass


## -------------------------------------------------------------------------------------------------
    def save(self, p_path, p_filename = 'summary.csv') -> bool:
        pass







## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class SLTraining (Training):
    """
    To be designed.
    """

    C_NAME = 'SL'


## -------------------------------------------------------------------------------------------------
    def __init__(self, **p_kwargs):

        Training.__init__(self)

        raise NotImplementedError


## -------------------------------------------------------------------------------------------------
    def _init_results(self) -> TrainingResults:

        results = Training._init_results(self)

        # If collect mappings, then create an input, target and output data logging object
        #     For this datalogger please get the input out put space and add dimensions
        #     accordingly to the data logger object

        # Add the rest three epochs based on conditions
        # Connect the training datalogger
        # If there is an evaluation dataset and test dataset
        # connect rest three data logging objects


        # connect the similar data logger to the scenario class

        return results


## -------------------------------------------------------------------------------------------------
    def _run_cycle(self) -> bool:
        # Set eof_epoch and eof_training to False

        # Initiate a new epoch if the cycles in a new epoch is set to 0 in a different place

        # get results from a cycle run
        # add number to the num epochs

        # if adapted is true add a number to num adaptations

        # if mode is evaluation update the evaluation

        # if mode is train update the epoch

        # get last loss from the model abd also get the last learning rate of the model

        # test if this is the end of epoch if the dataset counter has reached
        # to the last deliverable item in the dataset

        # If this is the end of the epoch then log it in the training console, based on what is the reason for the end of the epoch

        # if this is eof close the epoch and do the rituals

        # check if the adaptation limit is reached, and close the training if the adaptation limit is reached

        # return end of training

        pass


## -------------------------------------------------------------------------------------------------
    def _init_epoch(self):

        # check if the epoch needs to initiated in training mode or in the evaluation mode
        # if self._eval_dataset is not None:
        #     if self._mode is Train
        # if this is the first cycle in a new epoch, turn on the adaptivity
        # do the logging

        # if the mode is eval and, it's the first cycle in a new eval epoch
        #    Turn off the adaptivity of the model
        #    Initiate the evaluation epoch
        #    do the logging

        # But if there is no evaluation dataset
        # just do the logging normally, no need to take care of the adaptivity

        #The initialization still needs adding epoch to the corresponding datastoring object
        # In this case these would be, train, eval and validation
        # There would be another data storing object that will store the input, target and output

        pass


## -------------------------------------------------------------------------------------------------
    def setup_dataset(self):
        # get the dataset setup config, and call the split method of the dataset with names to the splitted datasets
        # and assign the returned dataset to self._dataset
        pass


## -------------------------------------------------------------------------------------------------
    def _update_epoch(self):
        #Not required at the moment
        # For updating the scores for the epoch
        pass


## -------------------------------------------------------------------------------------------------
    def _close_epoch(self):
        # Check if the evaluation dataset is there,
        # then you must handle the evalutation epochs before closing the training epochs

        # If the mode is training
        #    Logging for training epoch finished
        #    Increase the num epochs in the results

        #    If the mode is train change it to evaluation
        #       set the step in the particular evaluation epoch to +0

        # If the mode is to evaluation
        #     Logging for the epoch finished
        #     Increase the evaluation cycles by +1
        #     Calculate the scores in case of end of evaluation, calling the method close evaluation
        #     assign the high-score and corresponding model and epoch to the results

        #     If the mode is eval and the cycle limit is not reached, change it to train
        #     update the evaluation epoch +1
        #     update the train step to 0

        # If you dont have to handle the evaluation
        # just log everything and increase the number of epochs by +1
        # Reset the step counter to 0
        pass


## -------------------------------------------------------------------------------------------------
    def _init_eval(self):
        # Change the data loggers to the evaluation data loggers

        # change the particular evaluation cycles to 0

        pass


## -------------------------------------------------------------------------------------------------
    def _update_eval(self):
        # Update the evaluation
        # Calculate moving averages
        pass


## -------------------------------------------------------------------------------------------------
    def _close_eval(self):

        pass


## -------------------------------------------------------------------------------------------------
    def add_metric(self):
        pass



