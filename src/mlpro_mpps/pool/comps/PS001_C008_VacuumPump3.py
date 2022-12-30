## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - A Synoptic Framework for Standardized Machine Learning Tasks
## -- Package : mlpro_mpps.pool.comps
## -- Module  : PS001_C008_VacuumPump3.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2022-12-30  0.0.0     SY       Creation
## -- 2022-12-30  1.0.0     SY       Release of first version
## -------------------------------------------------------------------------------------------------

"""
Ver. 1.0.0 (2022-12-30)

This module provides a default implementation of a component of the BGLP, which is a Vacuum Pump.
This vacuum pump is located on Module 4 of the BGLP to transport materials from Hopper C to
finished good inventory with a constant outflow.
"""


from mlpro_mpps.mpps import *
from mlpro_at_basis.bf import *
from mlpro.bf.math import *
import sys




## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class VC3TransportedMaterial(SimState):
    """
    This class serves as a component state to calculate the transported material.
    """

    C_TYPE = 'SimState'
    C_NAME = 'VC3TransportedMaterial'
  
    
## -------------------------------------------------------------------------------------------------      
    def setup_function(self) -> TransferFunction:
        _func = TF_ConstVacuumPump(p_name='TF_ConstVacuumPump',
                                   p_type=TransferFunction.C_TRF_FUNC_CUSTOM,
                                   p_dt=0.05,
                                   prod_target=0.1)
        return _func


                 
                        
## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class TF_ConstVacuumPump(TransferFunction):
  
    
## -------------------------------------------------------------------------------------------------      
    def set_function_parameters(self, p_args) -> bool:
        if self.get_type() == self.C_TRF_FUNC_CUSTOM:
            try:
                self.prod_target = p_args['prod_target']
            except:
                raise NotImplementedError('One/More parameters for this function is missing.')           
        return True
  
    
## -------------------------------------------------------------------------------------------------      
    def custom_function(self, p_input, p_range=None):
        """
        To measure the transported material.

        Parameters
        ----------
        p_input : list
            [0] = Status of the actuator
        p_range : float
            period of measuring the transported material in seconds.

        Returns
        -------
        float
            The transported material.
        """
        if self.p_input[0]:
            if p_range is None:
                mass_transport = self.prod_target
            else:
                mass_transport = self.prod_target*p_range
            return mass_transport
        else:
            return 0


                 
                        
## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class VacuumPump3(Component):


## -------------------------------------------------------------------------------------------------
    def setup_component(self):
        """
        A vacuum pump consists of an actuator and two states components.
        """
        switch = SimActuator(p_name_short='Switch',
                             p_base_set=Dimension.C_BASE_SET_Z,
                             p_unit='s',
                             p_boundaries=[0, 1])
        transported_material = VC3TransportedMaterial(p_name_short='VC1TransportedMaterial',
                                                      p_base_set=Dimension.C_BASE_SET_R,
                                                      p_unit='L',
                                                      p_boundaries=[0,sys.maximize])
        
        self.add_actuator(p_actuator=switch)
        self.add_component_states(p_comp_states=transported_material)