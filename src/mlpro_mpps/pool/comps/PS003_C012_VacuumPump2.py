## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - A Synoptic Framework for Standardized Machine Learning Tasks
## -- Package : mlpro_mpps.pool.comps
## -- Module  : PS003_C012_VacuumPump2.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2023-11-12  0.0.0     SY       Creation
## -- 2023-11-12  1.0.0     SY       Release of first version
## -------------------------------------------------------------------------------------------------

"""
Ver. 1.0.0 (2023-11-12)

This module provides a default implementation of a component of the LS-BGLP, which is a Vacuum Pump
Type 2.
"""


from mlpro_mpps.mpps import *
from mlpro.bf.physics import TransferFunction
from mlpro.bf.math import *
from mlpro_mpps.pool.comps.PS003_C011_VacuumPump1 import *
import sys


                     
                        
## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class VC2_TransportedMaterial(SimState):
    """
    This class serves as a component state to calculate the transported material.
    """

    C_TYPE = 'SimState'
    C_NAME = 'VC2_TransportedMaterial'
  
    
## -------------------------------------------------------------------------------------------------      
    def _setup_function(self) -> TransferFunction:
        _func = TF_VacuumPump(p_name='TF_VacuumPump2',
                              p_type=TransferFunction.C_TRF_FUNC_CUSTOM,
                              p_dt=0.05,
                              coef=[0.3175, 0.0332])
        return _func


                     
                        
## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class VC2_PowerConsumption(SimState):
    """
    This class serves as a component state to calculate the power consumption.
    """

    C_TYPE = 'SimState'
    C_NAME = 'VC2_PowerConsumption'
  
    
## -------------------------------------------------------------------------------------------------      
    def _setup_function(self) -> TransferFunction:
        _func = TF_PowerVacuumPump(p_name='TF_PowerVacuumPump2',
                                   p_type=TransferFunction.C_TRF_FUNC_CUSTOM,
                                   p_dt=0.05,
                                   min_power=0,
                                   max_power=305,
                                   min_duration=0.567,
                                   max_duration=4.575)
        return _func


                 
                        
## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class VacuumPump2(Component):


## -------------------------------------------------------------------------------------------------
    def _setup_component(self):
        """
        A vacuum pump consists of an actuator and two states components.
        """
        timer = SimActuator(p_name_short='Timer',
                            p_base_set=Dimension.C_BASE_SET_R,
                            p_unit='s',
                            p_boundaries=[0.567, 4.575])
        transported_material = VC2_TransportedMaterial(p_name_short='VC2_TransportedMaterial',
                                                       p_base_set=Dimension.C_BASE_SET_R,
                                                       p_unit='L',
                                                       p_boundaries=[0,sys.maxsize])
        power_consumption = VC2_PowerConsumption(p_name_short='VC2_PowerConsumption',
                                                 p_base_set=Dimension.C_BASE_SET_R,
                                                 p_unit='kW',
                                                 p_boundaries=[0,sys.maxsize])
        
        self._add_actuator(p_actuator=timer)
        self._add_component_states(p_comp_states=transported_material)
        self._add_component_states(p_comp_states=power_consumption)
    
