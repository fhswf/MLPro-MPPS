## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - A Synoptic Framework for Standardized Machine Learning Tasks
## -- Package : mlpro_mpps.pool.mpps
## -- Module  : PS001_BGLP.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2022-12-30  0.0.0     SY       Creation
## -- 2022-12-30  1.0.0     SY       Release of first version
## -- 2023-01-11  1.0.1     SY       Debugging on setup_mpps, adjusting sensors' indices
## -------------------------------------------------------------------------------------------------

"""
Ver. 1.0.1 (2023-01-11)

This module provides a default implementation of the BGLP in MLPro-MPPS.
"""


from mlpro_mpps.mpps import *
from mlpro_mpps.pool.mods.PS001_M001_Loading import *
from mlpro_mpps.pool.mods.PS001_M002_Storing import *
from mlpro_mpps.pool.mods.PS001_M003_Weighing import *
from mlpro_mpps.pool.mods.PS001_M004_Filling import *


                     
                        
## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class BGLP(SimMPPS):


## -------------------------------------------------------------------------------------------------
    def setup_mpps(self):
        
        # 0. Add reference
        self.C_SCIREF_TYPE    = self.C_SCIREF_TYPE_ARTICLE
        self.C_SCIREF_AUTHOR  = "Dorothea Schwung, Steve Yuwono, Andreas Schwung, Steven X. Ding"
        self.C_SCIREF_TITLE   = "Decentralized learning of energy optimal production policies using PLC-informed reinforcement learning"
        self.C_SCIREF_JOURNAL = "Computers & Chemical Engineering"
        self.C_SCIREF_YEAR    = "2021"
        self.C_SCIREF_MONTH   = "05"
        self.C_SCIREF_DAY     = "28"
        self.C_SCIREF_VOLUME  = "152"
        self.C_SCIREF_DOI     = "10.1016/j.compchemeng.2021.107382"
        
        # 1. Add elements
        loading = LoadingStation(p_name='LoadingStation')
        storing = StoringStation(p_name='StoringStation')
        weighing = WeighingStation(p_name='WeighingStation')
        filling = FillingStation(p_name='FillingStation')
        
        self.add_element(p_elem=loading)
        self.add_element(p_elem=storing)
        self.add_element(p_elem=weighing)
        self.add_element(p_elem=filling)

        # 2. Setup which actions connected to which actuators
        _actions_in_order = False

        # 3. Setup input signals for updating sensors or component states values
        _signals = []
        _sensors = self.get_sensors()
        _actuators = self.get_actuators()
        _comp_states = self.get_component_states()
        
        # 3.1. Actuators-related states
        _signals.append([_comp_states[4], _actuators[0].get_value, _actuators[0].get_status])
        _signals.append([_comp_states[5], _actuators[0].get_value, _actuators[0].get_status])
        _signals.append([_comp_states[10], _actuators[1].get_value, _actuators[1].get_status])
        _signals.append([_comp_states[11], _actuators[1].get_value, _actuators[1].get_status])
        _signals.append([_comp_states[12], _actuators[2].get_status])
        _signals.append([_comp_states[13], _actuators[2].get_status])
        _signals.append([_comp_states[18], _actuators[3].get_value, _actuators[3].get_status])
        _signals.append([_comp_states[19], _actuators[3].get_value, _actuators[3].get_status])
        _signals.append([_comp_states[20], _actuators[4].get_value, _actuators[4].get_status])
        _signals.append([_comp_states[21], _actuators[4].get_value, _actuators[4].get_status])
        _signals.append([_comp_states[23], _actuators[5].get_status])
        
        # 3.2. Buffers-related states
        _signals.append([_comp_states[0], _comp_states[0].get_value, _comp_states[4].get_value])
        _signals.append([_comp_states[1], _comp_states[0].get_value, _comp_states[4].get_value])
        _signals.append([_comp_states[2], _comp_states[2].get_value, _comp_states[4].get_value, _comp_states[10].get_value])
        _signals.append([_comp_states[3], _comp_states[2].get_value, _comp_states[4].get_value, _comp_states[10].get_value])
        _signals.append([_comp_states[6], _comp_states[6].get_value, _comp_states[10].get_value, _comp_states[12].get_value])
        _signals.append([_comp_states[7], _comp_states[6].get_value, _comp_states[10].get_value, _comp_states[12].get_value])
        _signals.append([_comp_states[8], _comp_states[8].get_value, _comp_states[12].get_value, _comp_states[18].get_value])
        _signals.append([_comp_states[9], _comp_states[8].get_value, _comp_states[12].get_value, _comp_states[18].get_value])
        _signals.append([_comp_states[14], _comp_states[14].get_value, _comp_states[18].get_value, _comp_states[20].get_value])
        _signals.append([_comp_states[15], _comp_states[14].get_value, _comp_states[18].get_value, _comp_states[20].get_value])
        _signals.append([_comp_states[16], _comp_states[16].get_value, _comp_states[20].get_value, _comp_states[23].get_value])
        _signals.append([_comp_states[17], _comp_states[16].get_value, _comp_states[20].get_value, _comp_states[23].get_value])
        _signals.append([_comp_states[22], _comp_states[22].get_value, _comp_states[23].get_value])        
                
        # 3.2. Buffers-related sensor
        _signals.append([_sensors[0], _comp_states[0].get_value])
        _signals.append([_sensors[1], _comp_states[0].get_value])
        _signals.append([_sensors[2], _comp_states[2].get_value])
        _signals.append([_sensors[3], _comp_states[4].get_value])
        _signals.append([_sensors[4], _comp_states[4].get_value])
        _signals.append([_sensors[5], _comp_states[6].get_value])
        _signals.append([_sensors[6], _comp_states[8].get_value])
        _signals.append([_sensors[7], _comp_states[8].get_value])
        _signals.append([_sensors[8], _comp_states[10].get_value])

        # 4. Return _actions_in_order and _signals
        return _actions_in_order, _signals



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
    
