## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - A Synoptic Framework for Standardized Machine Learning Tasks
## -- Package : mlpro_mpps.pool.comps
## -- Module  : PS003_C006_MixingSilo_17L.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2023-11-11  0.0.0     SY       Creation
## -- 2023-11-11  1.0.0     SY       Release of first version
## -------------------------------------------------------------------------------------------------

"""
Ver. 1.0.0 (2023-11-11)

This module provides a default implementation of a component of the BGLP, which is a 17.42L Mixing Silo.
A silo is a component to temporary store materials that consists of two sensors.
"""


from mlpro_mpps.mpps import *
from mlpro.bf.physics import TransferFunction
from mlpro.bf.math import *
from mlpro_mpps.pool.comps.PS003_C001_Silo_17L import *
import sys



         
            
## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class MixingSilo17_Sensor1(SimSensor):
    """
    This class serves as a sensor in the upper side of the silo to indicate whether the fill-level
    of the Silo closes to overflow or not.
    """

    C_TYPE = 'SimSensor'
    C_NAME = 'MixingSilo17_Sensor1'
  
    
## -------------------------------------------------------------------------------------------------      
    def _setup_function(self) -> TransferFunction:
        _func = TF_BufferSensor(p_name='TF_MixingSilo17_Sensor1',
                                p_type=TransferFunction.C_TRF_FUNC_CUSTOM,
                                p_dt=0,
                                theta=0.8*17.42) # 80% of the maximum fill-level
        return _func



                 
                    
## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class MixingSilo17_Sensor2(SimSensor):
    """
    This class serves as a sensor in the lower side of the silo to indicate whether the fill-level
    of the Silo closes to empty or not.
    """

    C_TYPE = 'SimSensor'
    C_NAME = 'MixingSilo17_Sensor2'
  
    
## -------------------------------------------------------------------------------------------------      
    def _setup_function(self) -> TransferFunction:
        _func = TF_BufferSensor(p_name='TF_MixingSilo17_Sensor2',
                                p_type=TransferFunction.C_TRF_FUNC_CUSTOM,
                                p_dt=0,
                                theta=0.2*17.42) # 20% of the maximum fill-level
        return _func


                     


## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class MixingSilo17_FillLevel(SimState):
    """
    This class serves as a component state to calculate the actual fill-level of the silo.
    """

    C_TYPE = 'SimState'
    C_NAME = 'MixingSilo17_FillLevel'
  
    
## -------------------------------------------------------------------------------------------------      
    def _setup_function(self) -> TransferFunction:
        _func = TF_FillLevel(p_name='TF_FillLevel_MixingSilo17',
                             p_type=TransferFunction.C_TRF_FUNC_CUSTOM,
                             p_dt=0,
                             max_vol=17.42,
                             min_vol=0)
        return _func


                     
                        
## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class MixingSilo17_Overflow(SimState):
    """
    This class serves as a component state to calculate the overflow level of the silo.
    """

    C_TYPE = 'SimState'
    C_NAME = 'MixingSilo17_Overflow'
  
    
## -------------------------------------------------------------------------------------------------      
    def _setup_function(self) -> TransferFunction:
        _func = TF_Overflow(p_name='TF_Overflow_MixingSilo17',
                            p_type=TransferFunction.C_TRF_FUNC_CUSTOM,
                            p_dt=0,
                            max_vol=17.42)
        return _func


                 
                        
## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class MixingSilo17(Component):


## -------------------------------------------------------------------------------------------------
    def _setup_component(self):
        """
        A silo consists of two sensors and two states components.
        """
        silo_sensor_1 = MixingSilo17_Sensor1(p_name_short='MixingSilo17_Sensor1',
                                             p_base_set=Dimension.C_BASE_SET_Z,
                                             p_boundaries=[0,1])
        silo_sensor_2 = MixingSilo17_Sensor2(p_name_short='MixingSilo17_Sensor2',
                                             p_base_set=Dimension.C_BASE_SET_Z,
                                             p_boundaries=[0,1])
        silo_fill_level = MixingSilo17_FillLevel(p_name_short='MixingSilo17_FillLevel',
                                                 p_base_set=Dimension.C_BASE_SET_R,
                                                 p_unit='L',
                                                 p_boundaries=[0,17.42])
        silo_overflow = MixingSilo17_Overflow(p_name_short='MixingSilo17_Overflow',
                                              p_base_set=Dimension.C_BASE_SET_R,
                                              p_unit='L',
                                              p_boundaries=[0,sys.maxsize])
        
        self._add_sensor(p_sensor=silo_sensor_1)
        self._add_sensor(p_sensor=silo_sensor_2)
        self._add_component_states(p_comp_states=silo_overflow)
        self._add_component_states(p_comp_states=silo_fill_level)
    
