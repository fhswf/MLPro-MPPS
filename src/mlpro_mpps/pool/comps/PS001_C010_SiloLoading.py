## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - A Synoptic Framework for Standardized Machine Learning Tasks
## -- Package : mlpro_mpps.pool.comps
## -- Module  : PS001_C010_SiloLoading.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2022-12-30  0.0.0     SY       Creation
## -- 2022-12-30  1.0.0     SY       Release of first version
## -- 2023-01-11  1.0.1     SY       Debugging (sys.maxsize related issue)
## -- 2023-01-16  1.0.2     SY       Change order between fill-level and overflow as comp. states
## -- 2023-01-18  1.0.3     SY       Update because TransferFunction is shifted to MLPro.bf.systems
## -- 2023-02-01  1.0.4     SY       Refactoring
## -- 2023-02-06  1.0.5     SY       Refactoring
## -------------------------------------------------------------------------------------------------

"""
Ver. 1.0.5 (2023-02-06)

This module provides a default implementation of a component of the BGLP, which is a Silo in a
Loading station with special mechanism.
A silo is a component to temporary store materials that consists of two sensors.
"""


from mlpro_mpps.mpps import *
from mlpro.bf.physics import TransferFunction
from mlpro.bf.math import *
from mlpro_mpps.pool.comps.PS001_C001_Silo import *
import sys





## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class SiloLoadingFillLevel(SimState):
    """
    This class serves as a component state to calculate the actual fill-level of the silo.
    """

    C_TYPE = 'SimState'
    C_NAME = 'SiloLoadingFillLevel'
  
    
## -------------------------------------------------------------------------------------------------      
    def _setup_function(self) -> TransferFunction:
        _func = TF_FillLevel_Loading(p_name='TF_FillLevel_Loading',
                                     p_type=TransferFunction.C_TRF_FUNC_CUSTOM,
                                     p_dt=0,
                                     max_vol = 17.42,
                                     min_vol = 0,
                                     theta_loading = 0.3*17.42)
        return _func


                 
                        
## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class TF_FillLevel_Loading(TransferFunction):
  
    
## -------------------------------------------------------------------------------------------------      
    def _set_function_parameters(self, p_args) -> bool:
        if self.get_type() == self.C_TRF_FUNC_CUSTOM:
            try:
                self.max_vol = p_args['max_vol']
                self.min_vol = p_args['min_vol']
                self.theta_loading = p_args['theta_loading']
            except:
                raise NotImplementedError('One/More parameters for this function is missing.')           
        return True
  
    
## -------------------------------------------------------------------------------------------------      
    def _custom_function(self, p_input, p_range=None):
        """
        To measure the current fill-level.

        Parameters
        ----------
        p_input : list
            [0] = Actual fill-level
            [1] = Volume out

        Returns
        -------
        float
            Actual fill-level.
        """
        output = p_input[0]-p_input[1]
        
        if output >= self.max_vol:
            return self.max_vol
        elif output <= self.theta_loading:
            return self.theta_loading
        elif output <= self.min_vol:
            return self.min_vol
        else:
            return output


                     
                        
## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class SiloLoadingOverflow(SimState):
    """
    This class serves as a component state to calculate the overflow level of the silo.
    """

    C_TYPE = 'SimState'
    C_NAME = 'SiloLoadingOverflow'
  
    
## -------------------------------------------------------------------------------------------------      
    def _setup_function(self) -> TransferFunction:
        _func = TF_Overflow_Loading(p_name='TF_Overflow_Loading',
                                    p_type=TransferFunction.C_TRF_FUNC_CUSTOM,
                                    p_dt=0,
                                    max_vol = 17.42,
                                    theta_loading = 0.3*17.42)
        return _func


                 
                        
## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class TF_Overflow_Loading(TransferFunction):
  
    
## -------------------------------------------------------------------------------------------------      
    def _set_function_parameters(self, p_args) -> bool:
        if self.get_type() == self.C_TRF_FUNC_CUSTOM:
            try:
                self.max_vol = p_args['max_vol']
                self.theta_loading = p_args['theta_loading']
            except:
                raise NotImplementedError('One/More parameters for this function is missing.')           
        return True
  
    
## -------------------------------------------------------------------------------------------------      
    def _custom_function(self, p_input, p_range=None):
        """
        To measure the current overflow level.

        Parameters
        ----------
        p_input : list
            [0] = Actual fill-level
            [1] = Volume out

        Returns
        -------
        float
            Actual fill-level.
        """
        cur_level = p_input[0]-p_input[1]
        
        if cur_level > self.max_vol:
            return cur_level-self.max_vol
        else:
            return 0


                 
                        
## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class SiloLoading(Component):


## -------------------------------------------------------------------------------------------------
    def _setup_component(self):
        """
        A silo consists of two sensors and two states components.
        """
        silo_sensor_1 = SiloSensor1(p_name_short='SiloSensor1',
                                    p_base_set=Dimension.C_BASE_SET_Z,
                                    p_boundaries=[0,1])
        silo_sensor_2 = SiloSensor2(p_name_short='SiloSensor2',
                                    p_base_set=Dimension.C_BASE_SET_Z,
                                    p_boundaries=[0,1])
        silo_fill_level = SiloLoadingFillLevel(p_name_short='SiloLoadingFillLevel',
                                               p_base_set=Dimension.C_BASE_SET_R,
                                               p_unit='L',
                                               p_boundaries=[0,17.42])
        silo_overflow = SiloLoadingOverflow(p_name_short='SiloLoadingOverflow',
                                            p_base_set=Dimension.C_BASE_SET_R,
                                            p_unit='L',
                                            p_boundaries=[0,sys.maxsize])
        
        self._add_sensor(p_sensor=silo_sensor_1)
        self._add_sensor(p_sensor=silo_sensor_2)
        self._add_component_states(p_comp_states=silo_overflow)
        self._add_component_states(p_comp_states=silo_fill_level)
    
