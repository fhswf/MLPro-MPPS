## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - A Synoptic Framework for Standardized Machine Learning Tasks
## -- Package : mlpro_mpps.examples
## -- Module  : howto_002_set_up_MPPS.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2023-02-13  0.0.0     SY       Creation
## -- 2023-02-13  1.0.0     SY       Release of first version
## -------------------------------------------------------------------------------------------------

"""
Ver. 1.0.0 (2023-02-13)

This example shows the procedure for setting up a system using MPPS.

You will learn:
    
    1) How to set up an MPPS class and add them with components and/or modules
    
    2) How to set up the workflow of MPPS
    
    3) How to set up a simulate reaction of MPPS
    
"""


from mlpro_mpps.mpps import *
from mlpro_mpps.examples.howto_001_set_up_components_and_modules_in_MPPS import *





class MySystemSimulation(SimMPPS):
    
    
    # 1. How to setup MPPS and its workflow
    def _setup_mpps(self, p_auto_adjust_names=True):
        
        # 1.1. Add reference (optional)
        self.C_SCIREF_TYPE    = self.C_SCIREF_TYPE_NONE
        
        # 1.2. Add elements (components/modules)
        station_1 = MyStation(p_name='station_1')
        station_2 = MyStation(p_name='station_2')
        
        self._add_element(p_elem=station_1)
        self._add_element(p_elem=station_2)
        
        # 1.3. Check duplications of the elements names
        while not self._elements_names_checker():
            if p_auto_adjust_names:
                self._elements_names_auto_adjust()
            else:
                raise NameError('There are duplications of the elements names. You can just simply set p_auto_adjust_names to True.')
        
        # 1.4. Setup which actions connected to which actuators
        self._actions_in_order = False
        
        # 1.5. Setup input signals for updating sensors or component states values
        _signals = []
        _sens = self.get_sensors()
        _acts = self.get_actuators()
        _sts = self.get_component_states()
        
        self._add_signal(
            _sts['state_1'],              # p_updated_elem
            _acts['Switch'].get_value,    # p_input_fcts[0]
            _acts['Motor'].get_value,     # p_input_fcts[1]
            )
        
        self._add_signal(
            _sts['state_1_1'],
            _acts['Switch_1'].get_value,
            _acts['Motor_1'].get_value,
            )
        
        self._add_signal(
            _sts['sensor_1'],
            _acts['state_1'].get_value,
            )
        
        self._add_signal(
            _sts['sensor_1_1'],
            _acts['state_1_1'].get_value,
            )
        
    
    # 2. How to setup up simulate reaction in MPPS
    def _simulate_reaction(self, p_state: State, p_action: Action) -> State:
    
        # 2.1. Set values to actuators. This depends on the input space of the Action
        if self._actions_in_order:
            actions = Action.get_sorted_values()
            for idx, (_, acts) in enumerate(self.get_actuators().items()):
                acts.set_value(actions[idx])
        else:
            raise NotImplementedError
        
        # 2.2. Update values of the sensors and component states
        for sig in self._signals:
            if len(sig[1:]) == 1:
                input = sig[1]()
            else:
                input = []
                for x in range(len(sig[1:])):
                    input.append(sig[x+1]())
            sig[0].simulate(input, p_range=None)
    
        # 2.3. Return the resulted states in the form of State object
        raise NotImplementedError
        