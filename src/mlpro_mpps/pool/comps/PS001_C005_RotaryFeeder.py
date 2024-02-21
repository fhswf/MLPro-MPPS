## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - A Synoptic Framework for Standardized Machine Learning Tasks
## -- Package : mlpro_mpps.pool.comps
## -- Module  : PS001_C005_RotaryFeeder.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2022-12-29  0.0.0     SY       Creation
## -- 2022-12-29  1.0.0     SY       Release of first version
## -- 2023-01-11  1.0.1     SY       Debugging (sys.maxsize related issue)
## -- 2023-01-18  1.0.2     SY       Update because TransferFunction is shifted to MLPro.bf.systems
## -- 2023-02-01  1.0.3     SY       Refactoring
## -- 2023-02-06  1.0.4     SY       Refactoring
## -------------------------------------------------------------------------------------------------

"""
Ver. 1.0.4 (2023-02-06)

This module provides a default implementation of a component of the BGLP, which is a Rotary Feeder.
A rotary feeder is located on Module 3 of the BGLP to transport materials from Silo C to Hopper C.
"""


from mlpro_mpps.mpps import *
from mlpro.bf.physics import TransferFunction
from mlpro.bf.math import *
from mlpro_mpps.pool.comps.PS001_C003_ConveyorBelt import *
import sys


                     
                        
## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class RFTransportedMaterial(SimState):
    """
    This class serves as a component state to calculate the transported material.
    """

    C_TYPE = 'SimState'
    C_NAME = 'RFTransportedMaterial'
  
    
## -------------------------------------------------------------------------------------------------      
    def _setup_function(self) -> TransferFunction:
        _func = TF_TransBelt_Cont(p_name='TF_TransBelt_Cont',
                                  p_type=TransferFunction.C_TRF_FUNC_CUSTOM,
                                  p_dt=0.05,
                                  coef=0.01249/60)
        return _func


                     
                        
## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class RFPowerConsumption(SimState):
    """
    This class serves as a component state to calculate the power consumption.
    """

    C_TYPE = 'SimState'
    C_NAME = 'RFPowerConsumption'
  
    
## -------------------------------------------------------------------------------------------------      
    def _setup_function(self) -> TransferFunction:
        _func = TF_PowerBelt_Cont(p_name='TF_PowerBelt_Cont',
                                  p_type=TransferFunction.C_TRF_FUNC_CUSTOM,
                                  p_dt=0.05,
                                  min_power = 114.828,
                                  max_power = 370,
                                  min_rpm = 450,
                                  max_rpm = 1450)
        return _func


                 
                        
## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class RotaryFeeder(Component):


## -------------------------------------------------------------------------------------------------
    def _setup_component(self):
        """
        A rotary feeder consists of an actuator and two states components.
        """
        motor = SimActuator(p_name_short='Motor',
                            p_base_set=Dimension.C_BASE_SET_R,
                            p_unit='rpm',
                            p_boundaries=[450,1450])
        transported_material = RFTransportedMaterial(p_name_short='RFTransportedMaterial',
                                                     p_base_set=Dimension.C_BASE_SET_R,
                                                     p_unit='L',
                                                     p_boundaries=[0,sys.maxsize])
        power_consumption = RFPowerConsumption(p_name_short='RFPowerConsumption',
                                               p_base_set=Dimension.C_BASE_SET_R,
                                               p_unit='kW',
                                               p_boundaries=[0,sys.maxsize])
        
        self._add_actuator(p_actuator=motor)
        self._add_component_states(p_comp_states=transported_material)
        self._add_component_states(p_comp_states=power_consumption)
    
