## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - The integrative middleware framework for standardized machine learning
## -- Package : mlpro.rl.pool.sarsbuffer
## -- Module  : PrioritizedBuffer
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2021-09-22  0.0.0     WB       Creation
## -- 2021-09-22  1.0.0     WB       Added PrioritizedBuffer Class and PrioritizedBufferElement,
## --                                including the required SegmentTree data structure
## -- 2021-09-26  1.0.1     WB       Bug Fix 
## -- 2025-07-17  1.2.0     DA       Refactoring
## -------------------------------------------------------------------------------------------------

"""
Ver. 1.2.0 (2025-07-17) 

This module provides the Prioritized Buffer based on the reference.
"""


from typing import Callable
import random
import operator

import numpy as np

from mlpro.rl.models import SARSElement, SARSBuffer 



# Export list for public API
__all__ = [ 'PrioritizedBufferElement', 
            'PrioritizedBuffer', 
            'SegmentTree', 
            'SumSegmentTree', 
            'MinSegmentTree' ]




## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class PrioritizedBufferElement(SARSElement):
    """
    Element of a State-Action-Reward-Buffer.
    """
    pass


## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class PrioritizedBuffer(SARSBuffer):
    """
    Prioritized Sampling State-Action-Reward-Buffer in dictionary.
    """
    
    
## -------------------------------------------------------------------------------------------------
    def __init__(self, p_size=1, alpha: float=0.3, beta: float=1):
    
        """
        Parameters:
            p_size (int, optional): Buffer size. Defaults to 1.
            alpha (float, optional): Prioritization level. Defaults to 0.3
            beta (float, optional): Prioritization Control. Defaults to 1. Should be increased gradualy to 1 by the end of training.
        """
        assert alpha >= 0
        assert beta >= 0
        super().__init__(p_size=p_size)
        
        self.alpha = alpha
        self.beta = beta
        
        tree_capacity = 1
        while tree_capacity < self._size:
            tree_capacity *= 2
        
        self.sum_tree = SumSegmentTree(tree_capacity)
        self.min_tree = MinSegmentTree(tree_capacity)
        self.max_priority = 1.0
    
    
## -------------------------------------------------------------------------------------------------    
    def add_element(self, p_elem:PrioritizedBufferElement):
        """
        Add element to the buffer.

        Parameters:
            p_elem (BufferElement): Element of Buffer
        """
        super().add_element(p_elem)
        idx = len(self._data_buffer)-1
        self.sum_tree[idx] = self.max_priority**self.alpha
        self.min_tree[idx] = self.max_priority**self.alpha
    
    
## -------------------------------------------------------------------------------------------------
    def _gen_sample_ind(self, p_num:int) -> list:
        """
        Generate random indices from the buffer.

        Parameters:
            p_num (int): Number of sample

        Returns:
            List of incides
        """
        buffer_length = len(self._data_buffer)
        p_sum = self.sum_tree.sum(0, buffer_length-1)
        p_list_idx = []
        segment = p_sum / buffer_length
        for i in range(p_num):
            a = segment*i
            b = segment*(i+1)
            upperbound = random.uniform(a, b)
            idx = self.sum_tree.retrieve(upperbound)
            p_list_idx.append(idx)
        return p_list_idx
        

## -------------------------------------------------------------------------------------------------
    def _extract_rows(self, p_list_idx:list):
        """
        Extract the element in the buffer based on a
        list of indices.

        Parameters:
            p_list_idx (list): List of indices

        Returns:
            Samples in dictionary
        """
        rows = {}
        for key in self._data_buffer:
            rows[key] = [self._data_buffer[key][i] for i in p_list_idx]
        p_sample = []
        buffer_length = len(self._data_buffer)
        
        p_min = self.min_tree.min()/self.sum_tree.sum()
        max_weight = (p_min*buffer_length)**(-self.beta)
        for idx in p_list_idx:
            p_sample.append(self.sum_tree[idx]/self.sum_tree.sum())
        weights = (np.array(p_sample*buffer_length)**(-self.beta))/max_weight
        
        rows['weights'] = list(weights)
        rows['p_list_idx'] = p_list_idx
        
        return rows


## -------------------------------------------------------------------------------------------------
    def get_latest(self):
        """
        Returns latest buffered element. 
        """
        try:
            return self._extract_rows([len(self._data_buffer)-1])
        except:
            return None
        
        
## -------------------------------------------------------------------------------------------------
    def get_all(self):
        """
        Return all buffered elements.
        """
        p_list_idx = [i for i in range(len(self._data_buffer))]
        return self._extract_rows(p_list_idx)


## -------------------------------------------------------------------------------------------------
    def update_priorities(self, p_list_idx:list, priorities:np.ndarray):
        """
        Updates the priority tree.
        Needs to be called during each training step, utilising the element-wise calculated loss.
        """
        assert len(p_list_idx) == len(priorities)
        assert np.min(priorities) > 0 
        assert min(p_list_idx) >= 0
        assert max(p_list_idx) <= len(self._data_buffer)
        
        new_priorities = priorities**self.alpha
        for i in range(len(p_list_idx)):
            self.sum_tree[p_list_idx[i]] = new_priorities[i]
            self.min_tree[p_list_idx[i]] = new_priorities[i]
        
        self.max_priority = max(self.max_priority, np.max(new_priorities))


## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class SegmentTree:
    """ 
    Reference:
    https://github.com/openai/baselines/blob/master/baselines/common/segment_tree.py
    Attributes:
        capacity (int)
        tree (list)
        operation (function)
    """


## -------------------------------------------------------------------------------------------------
    def __init__(self, capacity: int, operation: Callable, init_value: float):
        assert (
            capacity > 0 and capacity & (capacity - 1) == 0
        ), "capacity must be positive and a power of 2."
        self.capacity = capacity
        self.tree = [init_value for _ in range(2 * capacity)]
        self.operation = operation
        
        
## -------------------------------------------------------------------------------------------------
    def _operate_helper(
        self, start: int, end: int, node: int, node_start: int, node_end: int
    ) -> float:
        """Returns result of operation in segment."""
        if start == node_start and end == node_end:
            return self.tree[node]
        mid = (node_start + node_end) // 2
        if end <= mid:
            return self._operate_helper(start, end, 2 * node, node_start, mid)
        else:
            if mid + 1 <= start:
                return self._operate_helper(start, end, 2 * node + 1, mid + 1, node_end)
            else:
                return self.operation(
                    self._operate_helper(start, mid, 2 * node, node_start, mid),
                    self._operate_helper(mid + 1, end, 2 * node + 1, mid + 1, node_end),
                )


## -------------------------------------------------------------------------------------------------
    def operate(self, start: int = 0, end: int = 0) -> float:
        """Returns result of applying 'self.operation'."""
        if end <= 0:
            end += self.capacity
        end -= 1

        return self._operate_helper(start, end, 1, 0, self.capacity - 1)


## -------------------------------------------------------------------------------------------------
    def __setitem__(self, idx: int, val: float):
        """Set value in tree."""
        idx += self.capacity
        self.tree[idx] = val

        idx //= 2
        while idx >= 1:
            self.tree[idx] = self.operation(self.tree[2 * idx], self.tree[2 * idx + 1])
            idx //= 2


## -------------------------------------------------------------------------------------------------
    def __getitem__(self, idx: int) -> float:
        """Get real value in leaf node of tree."""
        assert 0 <= idx < self.capacity

        return self.tree[self.capacity + idx]
        
        
## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class SumSegmentTree(SegmentTree):
    """ 
    Reference:
    https://github.com/openai/baselines/blob/master/baselines/common/segment_tree.py
    """


## -------------------------------------------------------------------------------------------------
    def __init__(self, capacity: int):
        super(SumSegmentTree, self).__init__(
            capacity=capacity, operation=operator.add, init_value=0.0
        )


## -------------------------------------------------------------------------------------------------
    def sum(self, start: int = 0, end: int = 0) -> float:
        """Returns arr[start] + ... + arr[end]."""
        return super(SumSegmentTree, self).operate(start, end)


## -------------------------------------------------------------------------------------------------
    def retrieve(self, upperbound: float) -> int:
        """Find the highest index `i` about upper bound in the tree"""
        # TODO: Check assert case and fix bug
        assert 0 <= upperbound <= self.sum() + 1e-5, "upperbound: {}".format(upperbound)

        idx = 1

        while idx < self.capacity:  # while non-leaf
            left = 2 * idx
            right = left + 1
            if self.tree[left] > upperbound:
                idx = 2 * idx
            else:
                upperbound -= self.tree[left]
                idx = right
        return idx - self.capacity
        
        
## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class MinSegmentTree(SegmentTree):
    """ 
    Reference:
    https://github.com/openai/baselines/blob/master/baselines/common/segment_tree.py
    """


## -------------------------------------------------------------------------------------------------
    def __init__(self, capacity: int):
        super(MinSegmentTree, self).__init__(
            capacity=capacity, operation=min, init_value=float("inf")
        )


## -------------------------------------------------------------------------------------------------
    def min(self, start: int = 0, end: int = 0) -> float:
        """Returns min(arr[start], ...,  arr[end])."""
        return super(MinSegmentTree, self).operate(start, end)