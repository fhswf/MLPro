## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - The integrative middleware framework for standardized machine learning
## -- Package : mlpro.oa.streams.tasks.anomalydetectors
## -- Module  : paga_detectors.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2023-06-08  0.0.0     SK       Creation
## -- 2023-09-12  1.0.0     SK       Release
## -- 2023-11-21  1.0.1     SK       Time Stamp update
## -- 2024-02-25  1.1.0     SK       Visualisation update
## -- 2024-04-10  1.2.0     DA/SK    Refactoring
## -------------------------------------------------------------------------------------------------

"""
Ver. 1.2.0 (2024-04-10)

This module provides templates for anomaly detection to be used in the context of online adaptivity.
"""

from mlpro.oa.streams.basics import *
from mlpro.oa.streams.tasks.anomalydetectors.anomalies import GroupAnomaly
from mlpro.oa.streams.tasks.anomalydetectors.basics import AnomalyDetector



## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class AnomalyDetectorPAGA(AnomalyDetector):
    """
    This is the base class for multivariate online anomaly detectors. It raises an event when an
    anomaly is detected. This class has the added functionality to differentiate between different
    types of anomalies.

    """

    C_NAME          = 'Anomaly Detector'

    C_PLOT_ACTIVE           = True
    C_PLOT_STANDALONE       = False

## -------------------------------------------------------------------------------------------------
    def __init__(self,
                 p_group_anomaly_det : bool = True,
                 p_name:str = None,
                 p_range_max = StreamTask.C_RANGE_THREAD,
                 p_ada : bool = True,
                 p_duplicate_data : bool = False,
                 p_visualize : bool = False,
                 p_logging=Log.C_LOG_ALL,
                 **p_kwargs):

        super().__init__(p_name = p_name,
                         p_range_max = p_range_max,
                         p_ada = p_ada,
                         p_duplicate_data = p_duplicate_data,
                         p_visualize = p_visualize,
                         p_logging = p_logging,
                         **p_kwargs)
        
        self.group_anomalies = []
        self.group_anomalies_instances = []
        self.group_ano_scores = []
        self.group_anomaly_det = p_group_anomaly_det


## -------------------------------------------------------------------------------------------------
    def _buffer_anomaly(self, p_anomaly):
        """
        Method to be used to add a new anomaly. Please use as part of your algorithm.

        Parameters
        ----------
        p_anomaly : Anomaly
            Anomaly object to be added.

        Returns
        -------
        p_anomaly : Anomaly
            Modified Anomaly object.
        """

        if self.group_anomaly_det:
            self.group_anomalies.append(p_anomaly)
            self.group_anomalies_instances.append(p_anomaly.get_instance()[-1])
            self.group_ano_scores.append(p_anomaly.get_ano_scores())

            if len(self.group_anomalies_instances) > 1:

                if int(p_anomaly.get_instance()[-1].get_id()) - 1 == int(self.group_anomalies_instances[-2].get_id()):

                    if len(self.group_anomalies_instances) == 3:

                        for i in range(2):
                            self.remove_anomaly(self.group_anomalies[i])
                        self._ano_id -= 2
                        anomaly = GroupAnomaly(p_instances=self.group_anomalies_instances,
                                               p_ano_scores=self.group_ano_scores, p_visualize=self._visualize,
                                               p_raising_object=self,
                                               p_det_time=str(p_anomaly.get_instance()[-1].get_tstamp()))
                        anomaly.set_id( p_id = self._get_next_anomaly_id() )
                        self._anomalies[anomaly.get_id()] = anomaly
                        self.group_anomalies = []
                        self.group_anomalies.append(anomaly)
                        return anomaly

                    elif len(self.group_anomalies_instances) > 3:
                        self.group_anomalies[0].set_instances(self.group_anomalies_instances, self.group_ano_scores)
                        self.group_anomalies.pop(-1)
                        return self.group_anomalies[0]
                        
                    else:
                        p_anomaly.set_id( p_id = self._get_next_anomaly_id() )
                        self._anomalies[p_anomaly.get_id()] = p_anomaly
                        return p_anomaly
                    
                else:
                    self.group_anomalies = []
                    self.group_anomalies_instances = []
                    self.group_ano_scores = []
                    self.group_anomalies.append(p_anomaly)
                    self.group_anomalies_instances.append(p_anomaly.get_instance()[-1])
                    self.group_ano_scores.append(p_anomaly.get_ano_scores())
                    p_anomaly.set_id( p_id = self._get_next_anomaly_id() )
                    self._anomalies[p_anomaly.get_id()] = p_anomaly
                    return p_anomaly
            else:
                p_anomaly.set_id( p_id = self._get_next_anomaly_id() )
                self._anomalies[p_anomaly.get_id()] = p_anomaly
                return p_anomaly
            
        else:
            p_anomaly.set_id( p_id = self._get_next_anomaly_id() )
            self._anomalies[p_anomaly.get_id()] = p_anomaly
            return p_anomaly
