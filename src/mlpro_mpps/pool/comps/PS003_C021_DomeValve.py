## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - A Synoptic Framework for Standardized Machine Learning Tasks
## -- Package : mlpro_mpps.pool.comps
## -- Module  : PS003_C021_DomeValve.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2023-11-12  0.0.0     SY       Creation
## -- 2023-11-12  1.0.0     SY       Release of first version
## -------------------------------------------------------------------------------------------------

"""
Ver. 1.0.0 (2023-11-12)

This module provides a default implementation of a component of the LS-BGLP, which is a Dome Valve.
"""


from mlpro_mpps.mpps import *
from mlpro.bf.physics import TransferFunction
from mlpro.bf.math import *
from mlpro_mpps.pool.comps.PS003_C019_VibratoryConveyor import *
import sys


                     
                        
## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class DV_TransportedMaterial(SimState):
    """
    This class serves as a component state to calculate the transported material.
    """

    C_TYPE = 'SimState'
    C_NAME = 'DV_TransportedMaterial'
  
    
## -------------------------------------------------------------------------------------------------      
    def _setup_function(self) -> TransferFunction:
        _func = TF_TransBelt_Binary(p_name='TF_TransDV_Binary',
                                    p_type=TransferFunction.C_TRF_FUNC_CUSTOM,
                                    p_dt=0.05,
                                    coef=0.325)
        return _func


                     
                        
## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class DV_PowerConsumption(SimState):
    """
    This class serves as a component state to calculate the power consumption.
    """

    C_TYPE = 'SimState'
    C_NAME = 'DV_PowerConsumption'
  
    
## -------------------------------------------------------------------------------------------------      
    def _setup_function(self) -> TransferFunction:
        _func = TF_PowerBelt_Binary(p_name='TF_PowerDV_Binary',
                                    p_type=TransferFunction.C_TRF_FUNC_CUSTOM,
                                    power=20.25)
        return _func


                 
                        
## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class DomeValve(Component):


## -------------------------------------------------------------------------------------------------
    def _setup_component(self):
        """
        A bucket elevator consists of an actuator and two states components.
        """
        switch = SimActuator(p_name_short='Switch',
                             p_base_set=Dimension.C_BASE_SET_Z,
                             p_unit='',
                             p_boundaries=[0,1])
        transported_material = DV_TransportedMaterial(p_name_short='DV_TransportedMaterial',
                                                      p_base_set=Dimension.C_BASE_SET_R,
                                                      p_unit='L',
                                                      p_boundaries=[0,sys.maxsize])
        power_consumption = DV_PowerConsumption(p_name_short='DV_PowerConsumption',
                                                p_base_set=Dimension.C_BASE_SET_R,
                                                p_unit='kW',
                                                p_boundaries=[0,sys.maxsize])
        
        self._add_actuator(p_actuator=switch)
        self._add_component_states(p_comp_states=transported_material)
        self._add_component_states(p_comp_states=power_consumption)
    
