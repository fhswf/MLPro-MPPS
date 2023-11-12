## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - A Synoptic Framework for Standardized Machine Learning Tasks
## -- Package : mlpro_mpps.pool.comps
## -- Module  : PS003_C009_ConveyorBelt1.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2023-11-12  0.0.0     SY       Creation
## -- 2023-11-12  1.0.0     SY       Release of first version
## -------------------------------------------------------------------------------------------------

"""
Ver. 1.0.0 (2023-11-12)

This module provides a default implementation of a component of the LS-BGLP, which is a Conveyor Belt
Type 1.
"""


from mlpro_mpps.mpps import *
from mlpro.bf.physics import TransferFunction
from mlpro.bf.math import *
import sys


                     
                        
## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class CB1_TransportedMaterial(SimState):
    """
    This class serves as a component state to calculate the transported material.
    """

    C_TYPE = 'SimState'
    C_NAME = 'CB1_TransportedMaterial'
  
    
## -------------------------------------------------------------------------------------------------      
    def _setup_function(self) -> TransferFunction:
        _func = TF_TransBelt1_Cont(p_name='TF_TransBelt1_Cont',
                                   p_type=TransferFunction.C_TRF_FUNC_CUSTOM,
                                   p_dt=0.05,
                                   coef=0.01/60)
        return _func


                 
                        
## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class TF_TransBelt1_Cont(TransferFunction):
  
    
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
class CB1_PowerConsumption(SimState):
    """
    This class serves as a component state to calculate the power consumption.
    """

    C_TYPE = 'SimState'
    C_NAME = 'CB1_PowerConsumption'
  
    
## -------------------------------------------------------------------------------------------------      
    def _setup_function(self) -> TransferFunction:
        _func = TF_PowerBelt1_Cont(p_name='TF_PowerBelt1_Cont',
                                   p_type=TransferFunction.C_TRF_FUNC_CUSTOM,
                                   p_dt=0.05,
                                   min_power=40.0,
                                   max_power=50.5,
                                   min_rpm=450,
                                   max_rpm=1800)
        return _func


                 
                        
## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class TF_PowerBelt1_Cont(TransferFunction):
  
    
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
class ConveyorBelt1(Component):


## -------------------------------------------------------------------------------------------------
    def _setup_component(self):
        """
        A conveyor belt consists of an actuator and two states components.
        """
        motor = SimActuator(p_name_short='Motor',
                            p_base_set=Dimension.C_BASE_SET_R,
                            p_unit='rpm',
                            p_boundaries=[450,1800])
        transported_material = CB1_TransportedMaterial(p_name_short='CB1_TransportedMaterial',
                                                       p_base_set=Dimension.C_BASE_SET_R,
                                                       p_unit='L',
                                                       p_boundaries=[0,sys.maxsize])
        power_consumption = CB1_PowerConsumption(p_name_short='CB1_PowerConsumption',
                                                 p_base_set=Dimension.C_BASE_SET_R,
                                                 p_unit='kW',
                                                 p_boundaries=[0,sys.maxsize])
        
        self._add_actuator(p_actuator=motor)
        self._add_component_states(p_comp_states=transported_material)
        self._add_component_states(p_comp_states=power_consumption)
    
