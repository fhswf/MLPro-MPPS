## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - A Synoptic Framework for Standardized Machine Learning Tasks
## -- Package : mlpro_mpps.pool.comps
## -- Module  : PS001_C003_ConveyorBelt.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2022-12-29  0.0.0     SY       Creation
## -- 2022-12-29  1.0.0     SY       Release of first version
## -- 2023-01-11  1.0.1     SY       Debugging (sys.maxsize related issue)
## -- 2023-01-13  1.0.2     SY       Debugging
## -- 2023-01-16  1.0.3     SY       Debugging on TF_PowerBelt_Cont
## -- 2023-01-18  1.0.3     SY       - Update because TransferFunction is shifted to MLPro.bf.systems
## --                                - Update transported material function
## -- 2023-02-01  1.0.4     SY       Refactoring
## -- 2023-02-06  1.0.5     SY       Refactoring
## -------------------------------------------------------------------------------------------------

"""
Ver. 1.0.5 (2023-02-06)

This module provides a default implementation of a component of the BGLP, which is a Conveyor Belt.
A conveyor belt is located on Module 1 of the BGLP to transport materials from Silo A to Hopper A.
"""


from mlpro_mpps.mpps import *
from mlpro.bf.physics import TransferFunction
from mlpro.bf.math import *
import sys


                     
                        
## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class CBTransportedMaterial(SimState):
    """
    This class serves as a component state to calculate the transported material.
    """

    C_TYPE = 'SimState'
    C_NAME = 'CBTransportedMaterial'
  
    
## -------------------------------------------------------------------------------------------------      
    def _setup_function(self) -> TransferFunction:
        _func = TF_TransBelt_Cont(p_name='TF_TransBelt_Cont',
                                  p_type=TransferFunction.C_TRF_FUNC_CUSTOM,
                                  p_dt=0.05,
                                  coef=0.01/60)
        return _func


                 
                        
## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class TF_TransBelt_Cont(TransferFunction):
  
    
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
            [0] = Rotational speed in rpm
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
                mass_transport = self.coef*p_input[0]
            else:
                mass_transport = self.coef*p_input[0]*p_range
            
            if mass_transport > p_input[2]:
                return p_input[2]
            else:
                return mass_transport
        else:
            return 0


                     
                        
## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class CBPowerConsumption(SimState):
    """
    This class serves as a component state to calculate the power consumption.
    """

    C_TYPE = 'SimState'
    C_NAME = 'CBPowerConsumption'
  
    
## -------------------------------------------------------------------------------------------------      
    def _setup_function(self) -> TransferFunction:
        _func = TF_PowerBelt_Cont(p_name='TF_PowerBelt_Cont',
                                  p_type=TransferFunction.C_TRF_FUNC_CUSTOM,
                                  p_dt=0.05,
                                  min_power = 40.0,
                                  max_power = 50.5,
                                  min_rpm = 450,
                                  max_rpm = 1850)
        return _func


                 
                        
## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class TF_PowerBelt_Cont(TransferFunction):
  
    
## -------------------------------------------------------------------------------------------------      
    def _set_function_parameters(self, p_args) -> bool:
        if self.get_type() == self.C_TRF_FUNC_CUSTOM:
            try:
                self.min_power = p_args['min_power']
                self.max_power = p_args['max_power']
                self.min_rpm = p_args['min_rpm']
                self.max_rpm = p_args['max_rpm']
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
            [0] = Rotational speed in rpm
            [1] = Status of the actuator
        p_range : float
            period of measuring the power consumption in seconds.

        Returns
        -------
        float
            The power consumption in kW.
        """
        if p_input[1]:
            normalized_rpm = (p_input[0]-self.min_rpm)/(self.max_rpm-self.min_rpm)
            if p_range is None:
                power  = normalized_rpm*(self.max_power-self.min_power)+self.min_power
            else:
                power  = (normalized_rpm*(self.max_power-self.min_power)+self.min_power)*p_range
            return power/1000.0
        else:
            return 0


                 
                        
## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class ConveyorBelt(Component):


## -------------------------------------------------------------------------------------------------
    def _setup_component(self):
        """
        A conveyor belt consists of an actuator and two states components.
        """
        motor = SimActuator(p_name_short='Motor',
                            p_base_set=Dimension.C_BASE_SET_R,
                            p_unit='rpm',
                            p_boundaries=[450,1850])
        transported_material = CBTransportedMaterial(p_name_short='CBTransportedMaterial',
                                                     p_base_set=Dimension.C_BASE_SET_R,
                                                     p_unit='L',
                                                     p_boundaries=[0,sys.maxsize])
        power_consumption = CBPowerConsumption(p_name_short='CBPowerConsumption',
                                               p_base_set=Dimension.C_BASE_SET_R,
                                               p_unit='kW',
                                               p_boundaries=[0,sys.maxsize])
        
        self._add_actuator(p_actuator=motor)
        self._add_component_states(p_comp_states=transported_material)
        self._add_component_states(p_comp_states=power_consumption)
    
