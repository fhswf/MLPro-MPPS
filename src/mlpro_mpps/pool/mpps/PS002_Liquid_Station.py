## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - A Synoptic Framework for Standardized Machine Learning Tasks
## -- Package : mlpro_mpps.pool.mpps
## -- Module  : PS002_Liquid_Station.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2023-01-19  0.0.0     ML       Creation
## -- 2023-03-28  1.0.0     ML/SY    Release of first version
## -------------------------------------------------------------------------------------------------

"""
Ver. 1.0.0 (2023-03-28)

This module provides a default implementation of the Liquid Station in MLPro-MPPS.
"""


from mlpro_mpps.mpps import *
from mlpro_mpps.pool.mods.PS002_M001_Station import Station


                     
                        
## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class Liquid_Station(SimMPPS):


## -------------------------------------------------------------------------------------------------
    def _setup_mpps(self, p_auto_adjust_names=True):
        
        # 0. Add elements
        station = Station(p_name='LiquidStation')
        
        self._add_element(p_elem=station)


       # 1. Check duplications of the elements names
        while not self._elements_names_checker():
            if p_auto_adjust_names:
                self._elements_names_auto_adjust()
            else:
                raise NameError('There are duplications of the elements names. You can just simply set p_auto_adjust_names to True.')

        # 2. Setup which actions connected to which actuators
        self._actions_in_order = False

        # 3. Setup input signals for updating sensors or component states values
        _sens = self.get_sensors()
        _acts = self.get_actuators()
        _sts = self.get_component_states()
        
        # 3.1. Actuators-related states
        # transport pump 1
        self._add_signal(_sts['PC1TransportedMaterial'],             # calculate transport matirial Pump 1
                         _acts['Timer1'].get_value,                  # get timer value Pump 1
                         _acts['Timer1'].get_status,                 # get status Pump 1
                         )
        
        # energy pump 1 
        self._add_signal(_sts['PC1PowerConsumption'],                # calculate energy consumption Pump 1
                         _acts['Timer1'].get_value,                  # get timer value Pump 1
                         _acts['Timer1'].get_status,                 # get status Pump 1
                         )
        
        # transport pump 2
        self._add_signal(_sts['PC2TransportedMaterial'],             # calculate transport matirial pump 2
                         _acts['Timer2'].get_value,                  # get timer value pump 2
                         _acts['Timer2'].get_status,                 # get status pump 2
                         _sts['TankFillLevel'].get_value,            # get fill level tank
                         )
        
        # energy pump 2
        self._add_signal(_sts['PC2PowerConsumption'],                # calculate energy consumption Pump 2
                         _acts['Timer2'].get_value,                  # get timer value pump 2
                         _acts['Timer2'].get_status,                 # get status pump 2
                         )

        # transport pump 3
        self._add_signal(_sts['PC3TransportedMaterial'],             # calculate transport matirial pump 3
                         _acts['Timer3'].get_value,                  # get timer value pump 3
                         _acts['Timer3'].get_status,                 # get status pump 3
                         _sts['TankFillLevel'].get_value,            # get fill level tank
                         )
        
        # energy pump 3
        self._add_signal(_sts['PC3PowerConsumption'],                # calculate energy consumption Pump 3
                         _acts['Timer3'].get_value,                  # get timer value pump 3
                         _acts['Timer3'].get_status,                 # get status pump 3
                         )
        
         # tank overflow 
        self._add_signal(_sts['TankOverflow'],                      # calculate tank overflow
                         _sts['TankFillLevel'].get_value,           # get fill level tank
                         _sts['PC1TransportedMaterial'].get_value,  # get transport matirial pump 1
                         _sts['PC2TransportedMaterial'].get_value,  # get transport matirial pump 2
                         _sts['PC3TransportedMaterial'].get_value,  # get transport matirial pump 3
                         )

        # tank level
        self._add_signal(_sts['TankFillLevel'],                     # calculate tank fill level 
                         _sts['TankFillLevel'].get_value,           # get tank fill level
                         _sts['PC1TransportedMaterial'].get_value,  # get transport matirial pump 1
                         _sts['PC2TransportedMaterial'].get_value,  # get transport matirial pump 2
                         _sts['PC3TransportedMaterial'].get_value,  # get transport matirial pump 3
                         )

        # tank sensor 1
        self._add_signal(_sens['TankSensor1'],                      # calculate signal tank sensor 1
                         _sts['TankFillLevel'].get_value            # get tank fill level
                         )
        
        # tank sensor 2
        self._add_signal(_sens['TankSensor2'],                      # calculate signal tank sensor 2
                         _sts['TankFillLevel'].get_value            # get tank fill level
                         )
        
        # tank sensor 3
        self._add_signal(_sens['TankSensor3'],                      # calculate signal tank sensor 3
                         _sts['TankFillLevel'].get_value            # get tank fill level
                         )


## -------------------------------------------------------------------------------------------------
    def _simulate_reaction(self, p_state: State, p_action: Action) -> State:
        
        # 1. Set values to actuators
        if self._actions_in_order:
            actions = Action.get_sorted_values()
            for idx, acts in enumerate(self.get_actuators()):
                acts.set_value(actions[idx])
        else:
            raise NotImplementedError
        
        # 2. Update values of the sensors and component states
        for sig in self._signals:
            if len(sig[1:]) == 1:
                input = sig[1]()
            else:
                input = []
                for x in range(len(sig[1:])):
                    input.append(sig[x+1]())
            sig[0].simulate(input, p_range=None)

        # 3. Return the resulted states in the form of State object
        raise NotImplementedError
    
