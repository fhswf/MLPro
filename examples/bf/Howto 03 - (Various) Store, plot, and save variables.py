## -------------------------------------------------------------------------------------------------
## -- Project : FH-SWF Automation Technology - Common Code Base (CCB)
## -- Package : mlpro
## -- Module  : Howto 03 - (Various) Store, plot, and save variables
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2021-06-16  1.0.0     SY       Creation/Release
## -- 2021-06-21  1.1.0     SY       Adjustment to updated DataPlotting class
## -- 2021-07-01  1.2.0     SY       Adjustment due to extension in save and load data
## -- 2021-09-11  1.2.0     MRD      Change Header information to match our new library name
## -------------------------------------------------------------------------------------------------

"""
Ver. 1.2.0 (2021-07-01)

This module demonstrates how to store, plot, save and load variables.
"""


from mlpro.bf.various import *
import random


num_eps         = 10
num_cycles      = 10000
data_names      = ["reward","states_1","states_2","model_loss"]
data_printing   = {"reward":        [True,0,10],
                   "states_1":      [True,0,4],
                   "states_2":      [True,0,4],
                   "model_loss":    [True,0,-1]}

## 1. How to store data ##
mem = DataStoring(data_names)
for ep in range(num_eps):
    ep_id = ("ep. %s"%str(ep+1))
    mem.add_frame(ep_id)
    for i in range(num_cycles):
        mem.memorize("reward",ep_id,random.uniform(0+(ep*0.5),5+(ep*0.5)))
        mem.memorize("states_1",ep_id,random.uniform(2-(ep*0.2),4-(ep*0.2)))
        mem.memorize("states_2",ep_id,random.uniform(0+(ep*0.2),2+(ep*0.2)))
        mem.memorize("model_loss",ep_id,random.uniform(0.25-(ep*0.02),1-(ep*0.07)))

## 2. How to plot stored data ##        
mem_plot    = DataPlotting(mem, p_window=100, p_showing=True, p_printing=data_printing)
mem_plot.get_plots()

## 3. How to save plots and data in binary file (variables, classes, etc.) ##
path_save   = input("Input path_save : ")
#Do not include quote-unquote ("" or '' ) into target path name
mem_plot.save_plots(path_save, "pdf")
mem_plot.save(path_save, "plot_memory")
mem.save(path_save, "data_memory")
mem.save_data(path_save, "data_storage", "\t")

## 4. How to load data from binary file ##
path_load   = path_save
mem_load    = DataStoring.load(path_load, "data_memory")
print("Comparison :")
print("Original data                : %.5f"%mem.memory_dict["reward"]["ep. 1"][0])
print("Loaded data from binary file : %.5f"%mem_load.memory_dict["reward"]["ep. 1"][0])

## 5. How to load data from csv file ##
data_names = []
mem_from_csv = DataStoring(data_names)
mem_from_csv.load_data(path_load, "data_storage.csv", "\t")
print("Comparison :")
print("Original data             : %.5f"%mem.memory_dict["reward"]["ep. 1"][0])
print("Loaded data from csv file : %.5f"%mem_from_csv.memory_dict["reward"]["ep. 1"][0])

