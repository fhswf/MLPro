## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - A Synoptic Framework for Standardized Machine Learning Tasks
## -- Package : mlpro
## -- Module  : howto_bf_006_buffers.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2021-10-26  1.0.0     SY       Creation/Release
## -------------------------------------------------------------------------------------------------

"""
Ver. 1.0.0 (2021-10-26)

This module demonstrates how to use classes Buffer and BufferElement.
"""


from mlpro.bf.data import *
import random



if __name__ == "__main__":

    # 1. Instantiate a buffer with random sampling
    buffer     = BufferRnd(p_size=100)
    num_cycles = 150


    # 2 Generate random values and store them to the Buffer
    for i in range(num_cycles):
        
        # 2.1 Store the values and their names in a BufferElement
        buffer_element  =  BufferElement({"reward":random.uniform(-10,10),
                                        "actions":[random.uniform(0,1),random.uniform(0,1)]})
        
        # 2.2 Example: add value element in the developed BufferElement
        buffer_element.add_value_element(dict(accuracy=random.uniform(0,1)))
        
        # 2.3 Add the BufferElement into the Buffer
        buffer.add_element(buffer_element)
        print('Cycle : %.i'%int(i+1))
        
        # 2.4 Checking whether buffer is full or not
        if not buffer.is_full():
            print('Buffer is not full yet, keep collecting data!\n')
        else:
            print('Buffer is full, ready to use!')
            
            # 2.5 Get all data from the Buffer
            all_data = buffer.get_all()
            _actions            = all_data["actions"]
            _reward             = all_data["reward"]
            _accuracy           = all_data["accuracy"]
            
            # 2.6 Get sample data from the Buffer, you define your sampling strategy by
            # redifining method _gen_sample_ind(self, p_num:int)
            sample_data = buffer.get_sample(p_num=10)
            print('Get sample!\n')
            _actions_sample    = sample_data["actions"]
            _reward_sample     = sample_data["reward"]
            _accuracy_sample   = sample_data["accuracy"]

    # 3 To clear your buffer        
    if buffer is not None:
        buffer.clear()
        print('Buffer is cleared!')

