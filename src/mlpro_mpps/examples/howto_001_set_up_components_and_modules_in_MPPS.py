## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - A Synoptic Framework for Standardized Machine Learning Tasks
## -- Package : mlpro_mpps.examples
## -- Module  : howto_001_set_up_components_and_modules_in_MPPS.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2023-02-13  0.0.0     SY       Creation
## -- 2023-02-13  1.0.0     SY       Release of first version
## -------------------------------------------------------------------------------------------------

"""
Ver. 1.0.0 (2023-02-13)

This example shows the procedure for setting up components and modules in MPPS.

You will learn:
    
    1) How to set up a sensor class and its incorporation with transfer learning
    
    2) How to set up a component state class and its incorporation with transfer learning
    
    3) How to set up a component class and add elements into the component
    
    4) How to set up a module class and add components into the module
    
"""


from mlpro_mpps.mpps import *
from mlpro.bf.physics import TransferFunction
from mlpro.bf.math import *
import sys





# 1. How to create a Sensor

# 1.1. Define a sensor class
class MySensor(SimSensor):
    
    C_TYPE = 'SimSensor'
    C_NAME = 'MySensor'
    
    
    # 1.2. Set up a sensor by adding transfer function for simulation of the sensor
    def _setup_function(self) -> TransferFunction:
        _func = TF_BufferSensor(p_name='TF_MySensor',
                                p_type=TransferFunction.C_TRF_FUNC_CUSTOM,
                                p_dt=0,
                                theta = 0.5)
        return _func
    
    
# 1.3. Define the transfer function class, see https://mlpro.readthedocs.io/content/02_basic_functions/mlpro_bf/sub/layer3_application_support/physics/01_transferfunction.html
class TF_MySensor(TransferFunction):
    
    
    def _set_function_parameters(self, p_args) -> bool:
        if self.get_type() == self.C_TRF_FUNC_CUSTOM:
            try:
                self.theta = p_args['theta']
            except:
                raise NotImplementedError('One/More parameters for this function is missing.')           
        return True
  
    
    # 1.4. Set up the transfer function of what does trigger the change of the sensor's value and how to calculate the new value.
    def _custom_function(self, p_input, p_range=None):
        if p_input >= self.theta:
            return True
        else:
            return False





# 2. How to create a Component State

# 2.1. Define a state class
class State1(SimState):

    C_TYPE = 'SimState'
    C_NAME = 'State1'
    
    
    # 2.2. Set up a state by adding transfer function for simulation of the state
    def _setup_function(self) -> TransferFunction:
        _func = TF_TransBelt_Cont(p_name='TF_MyState',
                                  p_type=TransferFunction.C_TRF_FUNC_CUSTOM,
                                  p_dt=0.05,
                                  coef=0.01/60)
        return _func
    
    
# 2.3. Define the transfer function class
class TF_MyState(TransferFunction):
    
    
    def _set_function_parameters(self, p_args) -> bool:
        if self.get_type() == self.C_TRF_FUNC_CUSTOM:
            try:
                self.theta = p_args['coef']
            except:
                raise NotImplementedError('One/More parameters for this function is missing.')           
        return True
  
    
    # 2.4. Set up the transfer function of what does trigger the change of the state's value and how to calculate the new value.
    def _custom_function(self, p_input, p_range=None):
        if p_input[0]:
            return p_input[1]*self.coef
        else:
            return 0
    
    
    
    
    
# 3. How to create a Component
# 3.1. Define a component class
class MyComponent(Component):
    
    
    # 3.2. Set up a component
    def _setup_component(self):
        # 3.3. Define one or more actuators, sensors, and component states related to the component
        motor = SimActuator(p_name_short='Motor',
                            p_base_set=Dimension.C_BASE_SET_R,
                            p_unit='rpm',
                            p_boundaries=[0,800])
        switch = SimActuator(p_name_short='Switch',
                             p_base_set=Dimension.C_BASE_SET_Z,
                             p_unit='',
                             p_boundaries=[0,1])
        sensor_1 = MySensor(p_name_short='sensor_1',
                            p_base_set=Dimension.C_BASE_SET_Z,
                            p_unit='',
                            p_boundaries=[0,1])
        state_1 = State1(p_name_short='state_1',
                         p_base_set=Dimension.C_BASE_SET_R,
                         p_unit='',
                         p_boundaries=[0,sys.maxsize])
        
        # 3.4. Add the defined elements into the component
        self._add_actuator(p_actuator=motor)
        self._add_actuator(p_actuator=switch)
        self._add_sensor(p_sensor=sensor_1)
        self._add_component_states(p_comp_states=state_1)
    
    
    
    
    
# 4. How to create a Module (Optional). Both module and component can be directly install to the MPPS
# 4.1. Define a module class
class MyStation(Module):
    
    
    # 4.2. Set up a module
    def _setup_module(self):
        # 4.3. Define one or more components related to the module
        component_1 = MyComponent(p_name='component_1')
        component_2 = MyComponent(p_name='component_2')
        
        # 4.4. Add the defined components into the module
        self._add_component(p_component=component_1)
        self._add_component(p_component=component_2)
    