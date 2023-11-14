## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - A Synoptic Framework for Standardized Machine Learning Tasks
## -- Package : mlpro_mpps.pool.comps
## -- Module  : PS003_C029_VacuumPump1_SP.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2023-11-13  0.0.0     SY       Creation
## -- 2023-11-13  1.0.0     SY       Release of first version
## -------------------------------------------------------------------------------------------------

"""
Ver. 1.0.0 (2023-11-13)

This module provides a default implementation of a component of the LS-BGLP, which is a Vacuum Pump
Type 1 for serial-parallel processes.
"""


from mlpro_mpps.mpps import *
from mlpro.bf.physics import TransferFunction
from mlpro.bf.math import *
from mlpro_mpps.pool.comps.PS003_C011_VacuumPump1 import *
import sys


                     
                        
## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class VC1SP_TransportedMaterial(SimState):
    """
    This class serves as a component state to calculate the transported material.
    """

    C_TYPE = 'SimState'
    C_NAME = 'VC1SP_TransportedMaterial'
  
    
## -------------------------------------------------------------------------------------------------      
    def _setup_function(self) -> TransferFunction:
        _func = TF_VacuumPumpSP(p_name='TF_VacuumPump1SP',
                                p_type=TransferFunction.C_TRF_FUNC_CUSTOM,
                                p_dt=0.05,
                                coef=[0.3535, 0.0096])
        return _func


                 
                        
## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class TF_VacuumPumpSP(TransferFunction):
  
    
## -------------------------------------------------------------------------------------------------      
    def _set_function_parameters(self, p_args) -> bool:
        if self.get_type() == self.C_TRF_FUNC_CUSTOM:
            try:
                self.coef = p_args['coef']
            except:
                raise NotImplementedError('One/More parameters for this function is missing.')           
        return True
  
    
## -------------------------------------------------------------------------------------------------      
    def _custom_function(self, p_input, p_range=None):
        """
        To measure the transported material.

        Parameters
        ----------
        p_input : list
            [0] = Turn-on duration
            [1] = Status of the actuator
            [2] = Fill-level of the previous first buffer
            [3] = Fill-level of the previous second buffer
        p_range : float
            period of measuring the transported material in seconds.

        Returns
        -------
        float
            The transported material.
        """
        if p_input[1]:
            if p_range is None:
                mass_transport = (2*self.coef[1])+self.coef[0]
            else:
                if p_input[0] <= p_range:
                    mass_transport = ((2*self.coef[1])+self.coef[0])*p_input[0]
                else:
                    mass_transport = ((2*self.coef[1])+self.coef[0])*p_range

            transported_mass = 0

            if (mass_transport*0.5) > p_input[2]:
                transported_mass += p_input[2]
            else:
                transported_mass += (mass_transport*0.5)

            if (mass_transport*0.5) > p_input[3]:
                transported_mass += p_input[3]
            else:
                transported_mass += (mass_transport*0.5)

            return transported_mass
        else:
            return 0


                     
                        
## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class VC1SP_PowerConsumption(SimState):
    """
    This class serves as a component state to calculate the power consumption.
    """

    C_TYPE = 'SimState'
    C_NAME = 'VC1SP_PowerConsumption'
  
    
## -------------------------------------------------------------------------------------------------      
    def _setup_function(self) -> TransferFunction:
        _func = TF_PowerVacuumPump(p_name='TF_PowerVacuumPump1SP',
                                   p_type=TransferFunction.C_TRF_FUNC_CUSTOM,
                                   p_dt=0.05,
                                   min_power=0,
                                   max_power=456,
                                   min_duration=0.979,
                                   max_duration=9.5)
        return _func


                 
                        
## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class VacuumPump1SP(Component):


## -------------------------------------------------------------------------------------------------
    def _setup_component(self):
        """
        A vacuum pump consists of an actuator and two states components.
        """
        timer = SimActuator(p_name_short='Timer',
                            p_base_set=Dimension.C_BASE_SET_R,
                            p_unit='s',
                            p_boundaries=[0.979, 9.5])
        transported_material = VC1SP_TransportedMaterial(p_name_short='VC1SP_TransportedMaterial',
                                                         p_base_set=Dimension.C_BASE_SET_R,
                                                         p_unit='L',
                                                         p_boundaries=[0,sys.maxsize])
        power_consumption = VC1SP_PowerConsumption(p_name_short='VC1SP_PowerConsumption',
                                                   p_base_set=Dimension.C_BASE_SET_R,
                                                   p_unit='kW',
                                                   p_boundaries=[0,sys.maxsize])
        
        self._add_actuator(p_actuator=timer)
        self._add_component_states(p_comp_states=transported_material)
        self._add_component_states(p_comp_states=power_consumption)
    
