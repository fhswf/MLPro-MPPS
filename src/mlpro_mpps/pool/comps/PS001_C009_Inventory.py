## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - A Synoptic Framework for Standardized Machine Learning Tasks
## -- Package : mlpro_mpps.pool.comps
## -- Module  : PS001_C009_Inventory.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2022-12-30  0.0.0     SY       Creation
## -- 2022-12-30  1.0.0     SY       Release of first version
## -------------------------------------------------------------------------------------------------

"""
Ver. 1.0.0 (2022-12-30)

This module provides a default implementation of a component of the BGLP, which is a finished goods
inventory.
A finished good inventory is a component to store the finished goods before being sent to the
next process or packaged.
"""


from mlpro_mpps.mpps import *
from mlpro_at_basis.bf import *
from mlpro.bf.math import *
import sys


         
            
## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class InventoryLevel(SimState):
    """
    This class serves as a component state to calculate the actual level of the inventory.
    """

    C_TYPE = 'SimState'
    C_NAME = 'InventoryLevel'
  
    
## -------------------------------------------------------------------------------------------------      
    def setup_function(self) -> TransferFunction:
        _func = TF_InventoryLevel(p_name='TF_InventoryLevel',
                                  p_type=TransferFunction.C_TRF_FUNC_CUSTOM,
                                  p_dt=0)
        return _func


                 
                        
## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class TF_InventoryLevel(TransferFunction):
  
    
## -------------------------------------------------------------------------------------------------      
    def set_function_parameters(self, p_args) -> bool:
        if self.get_type() == self.C_TRF_FUNC_CUSTOM:
            pass          
        return True
  
    
## -------------------------------------------------------------------------------------------------      
    def custom_function(self, p_input, p_range=None):
        """
        To measure the current level.

        Parameters
        ----------
        p_input : list
            [0] = Current level
            [1] = Volume in
            [2] = Reset

        Returns
        -------
        float
            Actual level.
        """
        if p_input[2]:
            return 0
        else:
            return p_input[0]+p_input[1]


                 
                        
## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class FinishedGoodsInventory(Component):


## -------------------------------------------------------------------------------------------------
    def setup_component(self):
        """
        A silo consists of two sensors and two states components.
        """
        inventory_level = InventoryLevel(p_name_short='InventoryLevel',
                                        p_base_set=Dimension.C_BASE_SET_R,
                                        p_unit='L',
                                        p_boundaries=[0,sys.maximize])
        
        self.add_component_states(p_comp_states=inventory_level)
    
