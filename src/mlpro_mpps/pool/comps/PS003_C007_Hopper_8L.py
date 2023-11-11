## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - A Synoptic Framework for Standardized Machine Learning Tasks
## -- Package : mlpro_mpps.pool.comps
## -- Module  : PS003_C007_Hopper_8L.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2023-11-11  0.0.0     SY       Creation
## -- 2023-11-11  1.0.0     SY       Release of first version
## -------------------------------------------------------------------------------------------------

"""
Ver. 1.0.0 (2023-11-11)

This module provides a default implementation of a component of the BGLP, which is a 8L Mini Hopper.
A hopper is a component to temporary store materials that consists of a sensor.
"""


from mlpro_mpps.mpps import *
from mlpro.bf.physics import TransferFunction
from mlpro.bf.math import *
from mlpro_mpps.pool.comps.PS003_C001_Silo_17L import *
import sys


         
            
## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class Hopper8_Sensor(SimSensor):
    """
    This class serves as a sensor in the upper side of the hopper to indicate whether the fill-level
    of the hopper closes to overflow or not.
    """

    C_TYPE = 'SimSensor'
    C_NAME = 'Hopper8_Sensor'
  
    
## -------------------------------------------------------------------------------------------------      
    def _setup_function(self) -> TransferFunction:
        _func = TF_BufferSensor(p_name='TF_Hopper8_Sensor',
                                p_type=TransferFunction.C_TRF_FUNC_CUSTOM,
                                p_dt=0,
                                theta=0.8*8.0) # 80% of the maximum fill-level
        return _func


                     
                        
## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class Hopper8_FillLevel(SimState):
    """
    This class serves as a component state to calculate the actual fill-level of the hopper.
    """

    C_TYPE = 'SimState'
    C_NAME = 'Hopper8_FillLevel'
  
    
## -------------------------------------------------------------------------------------------------      
    def _setup_function(self) -> TransferFunction:
        _func = TF_FillLevel(p_name='TF_Hopper8_FillLevel',
                             p_type=TransferFunction.C_TRF_FUNC_CUSTOM,
                             p_dt=0,
                             max_vol=8.0,
                             min_vol=0)
        return _func


                     
                        
## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class Hopper8_Overflow(SimState):
    """
    This class serves as a component state to calculate the overflow level of the hopper.
    """

    C_TYPE = 'SimState'
    C_NAME = 'Hopper8_Overflow'
  
    
## -------------------------------------------------------------------------------------------------      
    def _setup_function(self) -> TransferFunction:
        _func = TF_Overflow(p_name='TF_Hopper8_Overflow',
                            p_type=TransferFunction.C_TRF_FUNC_CUSTOM,
                            p_dt=0,
                            max_vol=8.0)
        return _func


                 
                        
## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class Hopper8(Component):


## -------------------------------------------------------------------------------------------------
    def _setup_component(self):
        """
        A silo consists of two sensors and two states components.
        """
        hopper_sensor = Hopper8_Sensor(p_name_short='Hopper8_Sensor1',
                                       p_base_set=Dimension.C_BASE_SET_Z,
                                       p_boundaries=[0,1])
        hopper_fill_level = Hopper8_FillLevel(p_name_short='Hopper8_FillLevel',
                                              p_base_set=Dimension.C_BASE_SET_R,
                                              p_unit='L',
                                              p_boundaries=[0,8.0])
        hopper_overflow = Hopper8_Overflow(p_name_short='Hopper8_Overflow',
                                           p_base_set=Dimension.C_BASE_SET_R,
                                           p_unit='L',
                                           p_boundaries=[0,sys.maxsize])
        
        self._add_sensor(p_sensor=hopper_sensor)
        self._add_component_states(p_comp_states=hopper_overflow)
        self._add_component_states(p_comp_states=hopper_fill_level)
    
