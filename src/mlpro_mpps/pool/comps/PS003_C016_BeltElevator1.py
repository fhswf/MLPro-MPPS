## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - A Synoptic Framework for Standardized Machine Learning Tasks
## -- Package : mlpro_mpps.pool.comps
## -- Module  : PS003_C016_BeltElevator1.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2023-11-12  0.0.0     SY       Creation
## -- 2023-11-12  1.0.0     SY       Release of first version
## -------------------------------------------------------------------------------------------------

"""
Ver. 1.0.0 (2023-11-12)

This module provides a default implementation of a component of the LS-BGLP, which is a Belt Elevator
Type 1.
"""


from mlpro_mpps.mpps import *
from mlpro.bf.physics import TransferFunction
from mlpro.bf.math import *
from mlpro_mpps.pool.comps.PS003_C009_ConveyorBelt1 import *
import sys


                     
                        
## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class BE1_TransportedMaterial(SimState):
    """
    This class serves as a component state to calculate the transported material.
    """

    C_TYPE = 'SimState'
    C_NAME = 'BE1_TransportedMaterial'
  
    
## -------------------------------------------------------------------------------------------------      
    def _setup_function(self) -> TransferFunction:
        _func = TF_TransBelt_Cont(p_name='TF_TransBE1_Cont',
                                  p_type=TransferFunction.C_TRF_FUNC_CUSTOM,
                                  p_dt=0.05,
                                  coef=0.015/60)
        return _func


                     
                        
## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class BE1_PowerConsumption(SimState):
    """
    This class serves as a component state to calculate the power consumption.
    """

    C_TYPE = 'SimState'
    C_NAME = 'BE1_PowerConsumption'
  
    
## -------------------------------------------------------------------------------------------------      
    def _setup_function(self) -> TransferFunction:
        _func = TF_PowerBelt_Cont(p_name='TF_PowerBE1_Cont',
                                  p_type=TransferFunction.C_TRF_FUNC_CUSTOM,
                                  p_dt=0.05,
                                  min_power=15.00,
                                  max_power=30.70,
                                  min_rpm=300,
                                  max_rpm=1300)
        return _func


                 
                        
## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class BeltElevator1(Component):


## -------------------------------------------------------------------------------------------------
    def _setup_component(self):
        """
        A belt elevator consists of an actuator and two states components.
        """
        motor = SimActuator(p_name_short='Motor',
                            p_base_set=Dimension.C_BASE_SET_R,
                            p_unit='rpm',
                            p_boundaries=[300,1300])
        transported_material = BE1_TransportedMaterial(p_name_short='BE1_TransportedMaterial',
                                                       p_base_set=Dimension.C_BASE_SET_R,
                                                       p_unit='L',
                                                       p_boundaries=[0,sys.maxsize])
        power_consumption = BE1_PowerConsumption(p_name_short='BE1_PowerConsumption',
                                                 p_base_set=Dimension.C_BASE_SET_R,
                                                 p_unit='kW',
                                                 p_boundaries=[0,sys.maxsize])
        
        self._add_actuator(p_actuator=motor)
        self._add_component_states(p_comp_states=transported_material)
        self._add_component_states(p_comp_states=power_consumption)
    
