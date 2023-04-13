## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - A Synoptic Framework for Standardized Machine Learning Tasks
## -- Package : mlpro_mpps.pool.rl_environment
## -- Module  : RL002_LS.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2023-03-10  0.0.0     ML       Creation
## -- 2023-03-28  1.0.0     ML/SY    Release of first version
## -- 2023-04-13  1.0.1     SY       Debugging
## -------------------------------------------------------------------------------------------------

"""
Ver. 1.0.1 (2023-04-13)

This module provides a default implementation of the Liquid Station in MLPro-MPPS as RL Environment.
"""


from mlpro_mpps.pool.mpps.PS002_Liquid_Station import Liquid_Station
from mlpro.bf.math import *
from mlpro.rl.models import *


                     

                        
## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class LS4RL(Liquid_Station):


## -------------------------------------------------------------------------------------------------
    def __init__(self, p_name:str, p_id:int=None, p_logging=Log.C_LOG_ALL, **p_kwargs):
        super().__init__(p_name=p_name, 
                         p_id=p_id, 
                         p_logging=p_logging, 
                         p_kwargs=p_kwargs
                         )
        
        try:
            self.parent = p_kwargs['p_parent']
        except:
            raise NotImplementedError('Please input the parent class of this class as p_parent')
            


## -------------------------------------------------------------------------------------------------
    def _simulate_reaction(self, p_state: State, p_action: Action) -> State:
        
        # 1. Set actuator values
        action = []

        # 1.1 Get action values from MLPro action
        for agent_id in p_action.get_agent_ids():
            action_elem = p_action.get_elem(agent_id)
            for action_id in action_elem.get_dim_ids():
                action.append(action_elem.get_value(action_id))
        
        # 1.2 Write action values to actuators
        for idx, (_, acts) in enumerate(self.get_actuators().items()):
            if idx != len(self.get_actuators())-1:
                boundaries = acts.get_boundaries()
                final_action = action[idx]*(boundaries[1]-boundaries[0])+boundaries[0]
                acts.set_value(final_action)
            else:
                acts.set_value(True)   
        
        # 2. Update values of the sensors and component states
        # store old tank level
        init_tank_level = self.get_component_states()['TankFillLevel'].get_value()

        # 2.1 Compute states and signals
        for sig in self._signals:
            # combine input signals
            if len(sig[1:]) == 1:
                input = sig[1]()
            else:
                input = []
                for x in range(len(sig[1:])):
                    input.append(sig[x+1]())
            # simulate
            sig[0].simulate(input, p_range=self.parent.t_set)

        # 3. Return the resulted states in the form of State object
        self.parent._state = self.parent.get_states()
        self.parent._state.set_success(False)
        self.parent._state.set_broken(False)
        
        # compute time step
        self.parent.t += self.parent.t_set
        
        # compute transport, overflow, power
        transport = sum(self.parent.get_transport())
        overlfow = sum(self.parent.get_overflow())
        power = sum(self.parent.get_power())

        # compute tank level
        current_volume = self.get_component_states()['TankFillLevel'].get_value()
        
        # store parameters in data frame
        self.parent.data_storing.memorize("time",str(self.parent.data_frame), self.parent.t)
        self.parent.data_storing.memorize("transport",str(self.parent.data_frame), transport/self.parent.t_set)
        self.parent.data_storing.memorize("overflow",str(self.parent.data_frame), overlfow/self.parent.t_set)
        self.parent.data_storing.memorize("power",str(self.parent.data_frame), power/self.parent.t_set)
        self.parent.data_storing.memorize("level",str(self.parent.data_frame), current_volume)
        
        return self.parent._state


                     
                        
## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class LS_RLEnv(Environment):

    C_TYPE = 'Environment'
    C_NAME = 'MPPS-based Liquid Station - RL Environment'
    C_CYCLE_LIMIT = 0  # Recommended cycle limit for training episodes


## -------------------------------------------------------------------------------------------------
    def __init__(self,
                 p_reward_type=Reward.C_TYPE_EVERY_AGENT,
                 p_logging=Log.C_LOG_ALL,
                 t_set=10.0,
                 lr_transport=1.0,
                 lr_overflow=4.0,
                 lr_power=0.0010, 
                 max_transport=[5, 5, 5],
                 prod_scenario='continuous',
                 cycle_limit=0):

        #self.num_envs = 5                                                 # Number of internal sub-environments
        self.reward_type = p_reward_type
        
        super().__init__(p_mode = Mode.C_MODE_SIM, 
                         p_latency = None, 
                         p_fct_strans = LS4RL(p_name='Liquid_Station_RL',
                                              p_logging=p_logging,
                                              p_parent=self
                                              ), 
                         p_fct_reward = None, 
                         p_fct_success = None, 
                         p_fct_broken = None, 
                         p_visualize = False, 
                         p_logging = p_logging)
        
        self.C_SCIREF_TYPE    = self.C_SCIREF_TYPE_ARTICLE
        self.C_SCIREF_AUTHOR  = "Marlon Löppenberg, Andreas Schwung"
        self.C_SCIREF_TITLE   = "Self Optimisation and Automatic Code Generation by Evolutionary Algorithms in PLC based Controlling Processes"
        self.C_SCIREF_JOURNAL = ""
        self.C_SCIREF_YEAR    = ""
        self.C_SCIREF_MONTH   = ""
        self.C_SCIREF_DAY     = ""
        self.C_SCIREF_VOLUME  = ""
        self.C_SCIREF_DOI     = ""
        
        self.C_CYCLE_LIMIT = cycle_limit
        self.t = 0
        self.t_set = t_set
        self.lr_transport = lr_transport
        self.lr_overflow = lr_overflow
        self.lr_power = lr_power
        self.max_transport = max_transport
        self.prod_scenario = prod_scenario
        
        self.data_lists = ["time","transport","overflow","power", "level"]
        self.data_storing = DataStoring(self.data_lists)
        self.data_frame = None
        
        # tank overflow 
        self.set_overflow = ['TankOverflow',
                             ]
        
        # tank level
        self.set_fill_levels = ['TankFillLevel',
                                ]
        
        # pump liquid transport
        self.set_transport_liquid = ['PC1TransportedMaterial',
                                     'PC2TransportedMaterial',
                                     'PC3TransportedMaterial'
                                     ]
        
        # pump energy consumption
        self.set_power = ['PC1PowerConsumption',
                          'PC2PowerConsumption',
                          'PC3PowerConsumption',
                          ]
        
        self.reset()


## -------------------------------------------------------------------------------------------------
    @staticmethod
    def setup_spaces():
        state_space = ESpace()
        action_space = ESpace()

        # define state space
        state_space.add_dim(Dimension('R-1 Lvl', 'R', 'Res-1 Level of Tank', '', '', '', [0, 250]))
        
        # define action space
        action_space.add_dim(Dimension('A-1 Act', 'R', 'Act-1 Pump', '', '', '', [0, 1]))
        action_space.add_dim(Dimension('A-2 Act', 'R', 'Act-2 Pump', '', '', '', [0, 1]))
        action_space.add_dim(Dimension('A-3 Act', 'R', 'Act-3 Pump', '', '', '', [0, 1]))

        return state_space, action_space


## -------------------------------------------------------------------------------------------------
    def get_states(self) -> State:
        state = State(self._state_space)
        ids = state.get_dim_ids()
        
        for x in range(len(ids)):
            fill_level = self._fct_strans.get_component_states()[self.set_fill_levels[x]].get_value()
            boundaries = self._fct_strans.get_component_states()[self.set_fill_levels[x]].get_boundaries()
            norm_fill_level = (fill_level-boundaries[0])/(boundaries[1]-boundaries[0])
            state.set_value(ids[x], norm_fill_level) 
        return state

## -------------------------------------------------------------------------------------------------
    def get_transport(self, p_outflow=True) -> list:
        total_transport = []

        # only output pumps (pump 1 and pump 2) count as transport
        if p_outflow==True:

            for x in [1, 2]:
                transport = self._fct_strans.get_component_states()[self.set_transport_liquid[x]].get_value()
                total_transport.append(transport)

            return total_transport

        # all pumps count as transport
        else:
            
            for x in range(len(self.set_transport_liquid)):
                transport = self._fct_strans.get_component_states()[self.set_transport_liquid[x]].get_value()
                total_transport.append(transport)

            return total_transport




## -------------------------------------------------------------------------------------------------
    def get_overflow(self) -> list:
        total_overflow = []
        
        for x in range(len(self.set_fill_levels)):
            overflow = self._fct_strans.get_component_states()[self.set_overflow[x]].get_value()
            total_overflow.append(overflow)
        return total_overflow


## -------------------------------------------------------------------------------------------------
    def get_power(self) -> list:
        total_power = []
        
        for x in range(len(self.set_power)):
            power = self._fct_strans.get_component_states()[self.set_power[x]].get_value()
            total_power.append(power)
        return total_power


## -------------------------------------------------------------------------------------------------
    def _compute_reward(self, p_state_old: State = None, p_state_new: State = None) -> Reward:
        reward = Reward(self.reward_type)

        if self.reward_type == Reward.C_TYPE_OVERALL:
            r_overall = 0
            r_overall = r_overall + sum(self.calc_reward())
            reward.set_overall_reward(r_overall)
        
        elif self.reward_type == Reward.C_TYPE_EVERY_AGENT:
           for agent_id in self._last_action.get_agent_ids():
               r_reward = self.calc_reward(self._last_action.get_agent_ids().index(agent_id))
               reward.add_agent_reward(agent_id, r_reward)
               
        else:
           for agent_id in self._last_action.get_agent_ids():
                agent_action_elem = self._last_action.get_elem(agent_id)
                agent_action_ids = agent_action_elem.get_dim_ids()
                r_reward = self.calc_reward()
                action_idx = 0
                for action_id in agent_action_ids:
                    r_action = r_reward[action_idx]
                    action_idx += 1
                    reward.add_action_reward(agent_id, action_id, r_action)
                    
        return reward


## -------------------------------------------------------------------------------------------------
    def _compute_success(self, p_state: State) -> bool:        
        if self.prod_scenario == 'continuous':
            return False
        else:
            raise NotImplementedError


## -------------------------------------------------------------------------------------------------
    def _compute_broken(self, p_state: State) -> bool:
        return False


## -------------------------------------------------------------------------------------------------
    def _reset(self, p_seed=None) -> None:

        # set seed
        random.seed(p_seed)

        # deactivate all actuators
        for acts in self._fct_strans.get_actuators():
            self._fct_strans.get_actuators()[acts].deactivate()

        # deactivate all sensors
        for sens in self._fct_strans.get_sensors():
            self._fct_strans.get_sensors()[sens].deactivate()
            
        # init tank fill level 
        for st in range(len(self.set_fill_levels)):
            buffer = self._fct_strans.get_component_states()[self.set_fill_levels[st]]          # get tank 
            boundaries = buffer.get_boundaries()                                                # get boundaries
            levels_init = random.uniform(0,1)                                                   # init uniform distribution
            fill_level = levels_init*(boundaries[1]-boundaries[0])+boundaries[0]                # compute fill level
            buffer.set_value(fill_level)                                                        # set fill level
        
        
        self.t = 0                              # reset time set
        self._state = self.get_states()         # set state
        self._state.set_success(False)          # set success
        self._state.set_broken(False)           # set broken
        
        # counte frame
        if self.data_frame == None:         
            self.data_frame = 0
        else:
            self.data_frame += 1
        self.data_storing.add_frame(str(self.data_frame))
            

## -------------------------------------------------------------------------------------------------
    def calc_reward(self, p_agent_id):
        reward = None

        transport = self.get_transport()
        overlfow = self.get_overflow()
        power = self.get_power()

        # compute specific reward - Agent 1 - overflow and energy
        if p_agent_id == 0:

            reward = (1.0 / ( 1.0 + self.lr_overflow * overlfow[0])) + (1.0 / ( 1.0 + self.lr_power * power[0]))

        # compute specific reward - Agent 2 - transport and energy
        elif p_agent_id == 1:

            reward = 1.0 / ( 1.0 + self.lr_overflow * (self.max_transport[1] - transport[0])) + 1.0 / ( 1.0 + self.lr_power * power[1])

        # compute specific reward - Agent 3 - transport and energy
        elif p_agent_id == 2:
            reward = 1.0 / ( 1.0 + self.lr_overflow * (self.max_transport[2] - transport[1])) + 1.0 / ( 11.0 + self.lr_power * power[2])

        else:
            raise NotImplementedError
        
        return reward





