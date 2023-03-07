.. _target_bf_ml_train_and_tune:
Training and Tuning
===================

A template for training models in their defined context is also introduced at this level. In a broader sense, this 
also includes finding an optimal value assignment for their hyperparameters. In MLPro, the **Training** class defines 
standards for this. Although abstract at this level, it fully implements the hyperparameter tuning here. The basic 
concept pursued here envisages executing an ML scenario under defined conditions and allowing the model contained 
therein to learn.


**Persistence of Training Results**

At the end of the training, the training results are saved in the file system. In particular, the entire scenario is 
saved here for later operational use. This includes both the trained model and the context in the last state.


**Scoring**

One of the training results is the **highscore**. Its determination is of course heavily dependent on the type of 
learning and can therefore only be specified in higher layers of MLPro. In any case, however, it is basically a real 
number that allows a qualitative statement about the learning performance of the model in its scenario.


**Hyperparameter Tuning**

Hyperparameter tuning is an optional training function performed by its own **HyperParamTuner** class. In particular, 
it defines the **maximize** method, which maximizes the highscore of a designated training by varying the hyperparameters 
of the model it contains. The optimization itself is not performed natively by MLPro, but by third-party packages. To this purpose, MLPro provides 
wrappers for :ref:`Optuna <Wrapper Optuna>` and :ref:`Hyperopt <Wrapper Hyperopt>`.


**Cross Reference**

- :ref:`API Reference BF-ML <target_api_bf_ml>`
- :ref:`Wrapper for Optuna <Wrapper Optuna>`
- :ref:`Wrapper for Hyperopt <Wrapper Hyperopt>`