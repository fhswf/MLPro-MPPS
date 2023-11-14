## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - A Synoptic Framework for Standardized Machine Learning Tasks
## -- Package : mlpro_mpps.pool.comps
## -- Module  : PS003_C010_ConveyorBelt2.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2023-11-12  0.0.0     SY       Creation
## -- 2023-11-12  1.0.0     SY       Release of first version
## -------------------------------------------------------------------------------------------------

"""
Ver. 1.0.0 (2023-11-12)

This module provides a default implementation of a component of the LS-BGLP, which is a Conveyor Belt
Type 2.
"""


from mlpro_mpps.mpps import *
from mlpro.bf.physics import TransferFunction
from mlpro.bf.math import *
from mlpro_mpps.pool.comps.PS003_C009_ConveyorBelt1 import *
import sys


                     
                        
## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class CB2_TransportedMaterial(SimState):
    """
    This class serves as a component state to calculate the transported material.
    """

    C_TYPE = 'SimState'
    C_NAME = 'CB2_TransportedMaterial'
  
    
## -------------------------------------------------------------------------------------------------      
    def _setup_function(self) -> TransferFunction:
        _func = TF_TransBelt_Cont(p_name='TF_TransBelt2_Cont',
                                  p_type=TransferFunction.C_TRF_FUNC_CUSTOM,
                                  p_dt=0.05,
                                  coef=0.01/60)
        return _func


                     
                        
## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class CB2_PowerConsumption(SimState):
    """
    This class serves as a component state to calculate the power consumption.
    """

    C_TYPE = 'SimState'
    C_NAME = 'CB2_PowerConsumption'
  
    
## -------------------------------------------------------------------------------------------------      
    def _setup_function(self) -> TransferFunction:
        _func = TF_PowerBelt_Cont(p_name='TF_PowerBelt2_Cont',
                                  p_type=TransferFunction.C_TRF_FUNC_CUSTOM,
                                  p_dt=0.05,
                                  min_power=20.5,
                                  max_power=42.5,
                                  min_rpm=450,
                                  max_rpm=1500)
        return _func


                 
                        
## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class ConveyorBelt2(Component):


## -------------------------------------------------------------------------------------------------
    def _setup_component(self):
        """
        A conveyor belt consists of an actuator and two states components.
        """
        motor = SimActuator(p_name_short='Motor',
                            p_base_set=Dimension.C_BASE_SET_R,
                            p_unit='rpm',
                            p_boundaries=[450,1500])
        transported_material = CB2_TransportedMaterial(p_name_short='CB2_TransportedMaterial',
                                                       p_base_set=Dimension.C_BASE_SET_R,
                                                       p_unit='L',
                                                       p_boundaries=[0,sys.maxsize])
        power_consumption = CB2_PowerConsumption(p_name_short='CB2_PowerConsumption',
                                                 p_base_set=Dimension.C_BASE_SET_R,
                                                 p_unit='kW',
                                                 p_boundaries=[0,sys.maxsize])
        
        self._add_actuator(p_actuator=motor)
        self._add_component_states(p_comp_states=transported_material)
        self._add_component_states(p_comp_states=power_consumption)
    
