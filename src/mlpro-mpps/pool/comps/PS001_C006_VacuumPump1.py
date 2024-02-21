## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - A Synoptic Framework for Standardized Machine Learning Tasks
## -- Package : mlpro_mpps.pool.comps
## -- Module  : PS001_C006_VacuumPump1.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2022-12-29  0.0.0     SY       Creation
## -- 2022-12-29  1.0.0     SY       Release of first version
## -- 2023-01-11  1.0.1     SY       Debugging (sys.maxsize related issue)
## -- 2023-01-13  1.0.2     SY       Debugging
## -- 2023-01-18  1.0.3     SY       - Update because TransferFunction is shifted to MLPro.bf.systems
## --                                - Update transported material function
## -- 2023-02-01  1.0.4     SY       Refactoring
## -- 2023-02-06  1.0.5     SY       Refactoring
## -------------------------------------------------------------------------------------------------

"""
Ver. 1.0.5 (2023-02-06)

This module provides a default implementation of a component of the BGLP, which is a Vacuum Pump.
This vacuum pump is located on Module 2 of the BGLP to transport materials from Hopper A to Silo B.
"""


from mlpro_mpps.mpps import *
from mlpro.bf.physics import TransferFunction
from mlpro.bf.math import *
import sys


                     
                        
## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class VC1TransportedMaterial(SimState):
    """
    This class serves as a component state to calculate the transported material.
    """

    C_TYPE = 'SimState'
    C_NAME = 'VC1TransportedMaterial'
  
    
## -------------------------------------------------------------------------------------------------      
    def _setup_function(self) -> TransferFunction:
        _func = TF_VacuumPump(p_name='TF_VacuumPump',
                              p_type=TransferFunction.C_TRF_FUNC_CUSTOM,
                              p_dt=0.05,
                              coef=[0.464, 0.0332])
        return _func


                 
                        
## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class TF_VacuumPump(TransferFunction):
  
    
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
            [2] = Fill-level of the previous buffer
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
        
            if mass_transport > p_input[2]:
                return p_input[2]
            else:
                return mass_transport
        else:
            return 0


                     
                        
## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class VC1PowerConsumption(SimState):
    """
    This class serves as a component state to calculate the power consumption.
    """

    C_TYPE = 'SimState'
    C_NAME = 'VC1PowerConsumption'
  
    
## -------------------------------------------------------------------------------------------------      
    def _setup_function(self) -> TransferFunction:
        _func = TF_PowerVacuumPump(p_name='TF_PowerVacuumPump',
                                   p_type=TransferFunction.C_TRF_FUNC_CUSTOM,
                                   p_dt=0.05,
                                   min_power = 0,
                                   max_power = 305,
                                   min_duration = 0.567,
                                   max_duration = 4.575)
        return _func


                 
                        
## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class TF_PowerVacuumPump(TransferFunction):
  
    
## -------------------------------------------------------------------------------------------------      
    def _set_function_parameters(self, p_args) -> bool:
        if self.get_type() == self.C_TRF_FUNC_CUSTOM:
            try:
                self.min_power = p_args['min_power']
                self.max_power = p_args['max_power']
                self.min_duration = p_args['min_duration']
                self.max_duration = p_args['max_duration']
            except:
                raise NotImplementedError('One/More parameters for this function is missing.')           
        return True
  
    
## -------------------------------------------------------------------------------------------------      
    def _custom_function(self, p_input, p_range=None):
        """
        To measure the power consumption.

        Parameters
        ----------
        p_input : list
            [0] = Turn-on duration
            [1] = Status of the actuator
        p_range : float
            period of measuring the power consumption in seconds.

        Returns
        -------
        float
            The power consumption in kW.
        """
        if p_input[1]:
            if p_range is None:
                power = self.max_power
            else:
                if p_input[0] <= p_range:
                    power = self.max_power*p_input[0]
                else:
                    power = self.max_power*p_range
            return power/1000.0
        else:
            return 0


                 
                        
## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class VacuumPump1(Component):


## -------------------------------------------------------------------------------------------------
    def _setup_component(self):
        """
        A vacuum pump consists of an actuator and two states components.
        """
        timer = SimActuator(p_name_short='Timer',
                            p_base_set=Dimension.C_BASE_SET_R,
                            p_unit='s',
                            p_boundaries=[0.567, 4.575])
        transported_material = VC1TransportedMaterial(p_name_short='VC1TransportedMaterial',
                                                      p_base_set=Dimension.C_BASE_SET_R,
                                                      p_unit='L',
                                                      p_boundaries=[0,sys.maxsize])
        power_consumption = VC1PowerConsumption(p_name_short='VC1PowerConsumption',
                                                p_base_set=Dimension.C_BASE_SET_R,
                                                p_unit='kW',
                                                p_boundaries=[0,sys.maxsize])
        
        self._add_actuator(p_actuator=timer)
        self._add_component_states(p_comp_states=transported_material)
        self._add_component_states(p_comp_states=power_consumption)
    
