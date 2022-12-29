## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - A Synoptic Framework for Standardized Machine Learning Tasks
## -- Package : mlpro_mpps.pool.comps
## -- Module  : PS001_C007_VacuumPump2.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2022-12-29  0.0.0     SY       Creation
## -- 2022-12-29  1.0.0     SY       Release of first version
## -------------------------------------------------------------------------------------------------

"""
Ver. 1.0.0 (2022-12-29)

This module provides a default implementation of a component of the BGLP, which is a Vacuum Pump.
This vacuum pump is located on Module 3 of the BGLP to transport materials from Hopper B to Silo C.
"""


from mlpro_mpps.mpps import *
from mlpro_at_basis.bf import *
from mlpro.bf.math import *
from mlpro_mpps.pool.comps.PS001_C006_VacuumPump1 import *
import sys


                     
                        
## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class VC2TransportedMaterial(SimState):
    """
    This class serves as a component state to calculate the transported material.
    """

    C_TYPE = 'SimState'
    C_NAME = 'VC2TransportedMaterial'
  
    
## -------------------------------------------------------------------------------------------------      
    def setup_function(self) -> TransferFunction:
        _func = TF_VacuumPump(p_name='TF_VacuumPump',
                              p_type=TransferFunction.C_TRF_FUNC_CUSTOM,
                              p_dt=0.05,
                              coef=[0.3535, 0.0096])
        return _func


                     
                        
## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class VC2PowerConsumption(SimState):
    """
    This class serves as a component state to calculate the power consumption.
    """

    C_TYPE = 'SimState'
    C_NAME = 'VC2PowerConsumption'
  
    
## -------------------------------------------------------------------------------------------------      
    def setup_function(self) -> TransferFunction:
        _func = TF_PowerVacuumPump(p_name='TF_PowerVacuumPump',
                                   p_type=TransferFunction.C_TRF_FUNC_CUSTOM,
                                   p_dt=0.05,
                                   min_power = 0,
                                   max_power = 456,
                                   min_duration = 0.979,
                                   max_duration = 9.5)
        return _func


                 
                        
## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class VacuumPump2(Component):


## -------------------------------------------------------------------------------------------------
    def setup_component(self):
        """
        A vacuum pump consists of an actuator and two states components.
        """
        timer = SimActuator(p_name_short='Timer',
                            p_base_set=Dimension.C_BASE_SET_R,
                            p_unit='s',
                            p_boundaries=[0.979, 9.5])
        transported_material = VC2TransportedMaterial(p_name_short='VC2TransportedMaterial',
                                                      p_base_set=Dimension.C_BASE_SET_R,
                                                      p_unit='L',
                                                      p_boundaries=[0,sys.maximize])
        power_consumption = VC2PowerConsumption(p_name_short='VC2PowerConsumption',
                                                p_base_set=Dimension.C_BASE_SET_R,
                                                p_unit='kW',
                                                p_boundaries=[0,sys.maximize])
        
        self.add_actuator(p_actuator=timer)
        self.add_component_states(p_comp_states=transported_material)
        self.add_component_states(p_comp_states=power_consumption)
    