## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - A Synoptic Framework for Standardized Machine Learning Tasks
## -- Package : mlpro_mpps.pool.comps
## -- Module  : C002_PS001_Hopper.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2022-12-29  0.0.0     SY       Creation
## -- 2022-12-29  1.0.0     SY       Release of first version
## -------------------------------------------------------------------------------------------------

"""
Ver. 1.0.0 (2022-12-29)

This module provides a default implementation of a component of the BGLP, which is a Mini Hopper.
A hopper is a component to temporary store materials that consists of a sensor.
"""


from mlpro_mpps.mpps import *
from mlpro_at_basis.bf import *
from mlpro_mpps.pool.comps.C001_PS001_Silo import *


         
            
## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class HopperSensor(SimSensor):
    """
    This class serves as a sensor in the upper side of the hopper to indicate whether the fill-level
    of the hopper closes to overflow or not.
    """

    C_TYPE = 'SimSensor'
    C_NAME = 'HopperSensor'
  
    
## -------------------------------------------------------------------------------------------------      
    def setup_function(self) -> TransferFunction:
        _func = TF_BufferSensor(p_name='TF_HopperSensor',
                                p_type=TransferFunction.C_TRF_FUNC_CUSTOM,
                                p_dt=0,
                                theta = 0.8*9.1) # 80% of the maximum fill-level
        return _func


                     
                        
## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class HopperFillLevel(SimState):
    """
    This class serves as a component state to calculate the actual fill-level of the hopper.
    """

    C_TYPE = 'SimState'
    C_NAME = 'HopperFillLevel'
  
    
## -------------------------------------------------------------------------------------------------      
    def setup_function(self) -> TransferFunction:
        _func = TF_FillLevel(p_name='TF_FillLevel',
                             p_type=TransferFunction.C_TRF_FUNC_CUSTOM,
                             p_dt=0,
                             max_vol = 9.1,
                             min_vol = 0)
        return _func


                     
                        
## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class HopperOverflow(SimState):
    """
    This class serves as a component state to calculate the overflow level of the hopper.
    """

    C_TYPE = 'SimState'
    C_NAME = 'HopperOverflow'
  
    
## -------------------------------------------------------------------------------------------------      
    def setup_function(self) -> TransferFunction:
        _func = TF_Overflow(p_name='TF_Overflow',
                            p_type=TransferFunction.C_TRF_FUNC_CUSTOM,
                            p_dt=0,
                            max_vol = 9.1)
        return _func


                 
                        
## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class Hopper(Component):


## -------------------------------------------------------------------------------------------------
    def setup_component(self):
        """
        A silo consists of two sensors and two states components.
        """
        hopper_sensor = HopperSensor(p_name_short='SiloSensor1',
                                     p_base_set=Dimension.C_BASE_SET_Z,
                                     p_boundaries=[0,1])
        hopper_fill_level = HopperFillLevel(p_name_short='HopperFillLevel',
                                            p_base_set=Dimension.C_BASE_SET_R,
                                            p_unit='L',
                                            p_boundaries=[0,9.1])
        hopper_overflow = HopperOverflow(p_name_short='HopperOverflow',
                                         p_base_set=Dimension.C_BASE_SET_R,
                                         p_unit='L',
                                         p_boundaries=[0,100])
        
        self.add_sensor(p_sensor=hopper_sensor)
        self.add_component_states(p_comp_states=hopper_fill_level)
        self.add_component_states(p_comp_states=hopper_overflow)
    
