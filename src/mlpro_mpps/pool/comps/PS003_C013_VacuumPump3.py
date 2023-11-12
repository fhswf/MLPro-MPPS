## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - A Synoptic Framework for Standardized Machine Learning Tasks
## -- Package : mlpro_mpps.pool.comps
## -- Module  : PS003_C013_VacuumPump3.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2023-11-12  0.0.0     SY       Creation
## -- 2023-11-12  1.0.0     SY       Release of first version
## -------------------------------------------------------------------------------------------------

"""
Ver. 1.0.0 (2023-11-12)

This module provides a default implementation of a component of the LS-BGLP, which is a Vacuum Pump
Type 3.
"""


from mlpro_mpps.mpps import *
from mlpro.bf.physics import TransferFunction
from mlpro.bf.math import *
from mlpro_mpps.pool.comps.PS003_C011_VacuumPump1 import *
import sys


                     
                        
## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class VC3_TransportedMaterial(SimState):
    """
    This class serves as a component state to calculate the transported material.
    """

    C_TYPE = 'SimState'
    C_NAME = 'VC3_TransportedMaterial'
  
    
## -------------------------------------------------------------------------------------------------      
    def _setup_function(self) -> TransferFunction:
        _func = TF_VacuumPump(p_name='TF_VacuumPump3',
                              p_type=TransferFunction.C_TRF_FUNC_CUSTOM,
                              p_dt=0.05,
                              coef=[0.2075, 0.0780])
        return _func


                     
                        
## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class VC3_PowerConsumption(SimState):
    """
    This class serves as a component state to calculate the power consumption.
    """

    C_TYPE = 'SimState'
    C_NAME = 'VC3_PowerConsumption'
  
    
## -------------------------------------------------------------------------------------------------      
    def _setup_function(self) -> TransferFunction:
        _func = TF_PowerVacuumPump(p_name='TF_PowerVacuumPump3',
                                   p_type=TransferFunction.C_TRF_FUNC_CUSTOM,
                                   p_dt=0.05,
                                   min_power=0,
                                   max_power=612,
                                   min_duration=0.979,
                                   max_duration=9.5)
        return _func


                 
                        
## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class VacuumPump3(Component):


## -------------------------------------------------------------------------------------------------
    def _setup_component(self):
        """
        A vacuum pump consists of an actuator and two states components.
        """
        timer = SimActuator(p_name_short='Timer',
                            p_base_set=Dimension.C_BASE_SET_R,
                            p_unit='s',
                            p_boundaries=[0.979, 9.5])
        transported_material = VC3_TransportedMaterial(p_name_short='VC3_TransportedMaterial',
                                                       p_base_set=Dimension.C_BASE_SET_R,
                                                       p_unit='L',
                                                       p_boundaries=[0,sys.maxsize])
        power_consumption = VC3_PowerConsumption(p_name_short='VC3_PowerConsumption',
                                                 p_base_set=Dimension.C_BASE_SET_R,
                                                 p_unit='kW',
                                                 p_boundaries=[0,sys.maxsize])
        
        self._add_actuator(p_actuator=timer)
        self._add_component_states(p_comp_states=transported_material)
        self._add_component_states(p_comp_states=power_consumption)
    
