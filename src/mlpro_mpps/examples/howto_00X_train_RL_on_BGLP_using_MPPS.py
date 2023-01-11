## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - A Synoptic Framework for Standardized Machine Learning Tasks
## -- Package : mlpro_mpps.examples
## -- Module  : howto_00X_train_RL_on_BGLP_using_MPPS.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2023-01-03  0.0.0     SY       Creation
## -- 2023-XX-XX  1.0.0     SY       Release of first version
## -------------------------------------------------------------------------------------------------

"""
Ver. 1.0.0 (2023-XX-XX)

This example shows the implementation of the MPPS-based BGLP as an RL Environment.
"""


from mlpro_mpps.pool.mpps.PS001_bglp import BGLP
from mlpro.bf.math import *
from mlpro.rl.models import *
from mlpro.rl.pool.policies.randomgenerator import RandomGenerator
import random
from pathlib import Path


                     
                        
## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class BGLP4RL(BGLP):


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
    def _simulate_reaction(self, p_state: State, p_action: Action) -> State:
        
        # 1. Set values to actuators
        action = []
        for agent_id in p_action.get_agent_ids():
            action_elem = p_action.get_elem(agent_id)
            for action_id in action_elem.get_dim_ids():
                action.append(action_elem.get_value(action_id))
                
        for idx, acts in enumerate(self.get_actuators()):
            if idx != len(self.get_actuators())-1:
                boundaries = acts.get_boundaries()
                final_action = action[idx]/(boundaries[1]-boundaries[0])+boundaries[0]
                acts.set_value(final_action)
        
        # 2. Update values of the sensors and component states
        self.init_inventory_level = self.get_component_states()[22].get_value()
        for sig in self._signals:
            if len(sig[1:]) == 1:
                input = sig[1]()
            else:
                input = []
                for x in range(len(sig[1:])):
                    input.append(sig[x+1]())
            sig[0].simulate(input, p_range=self.t_set)

        # 3. Return the resulted states in the form of State object
        self.set_actions(action)
        self._state = self.get_states()
        self._state.set_success(False)
        self._state.set_broken(False)
        
        self.t += self.t_set
        overlfow = self.get_overflow()
        power = self.get_power()
        demand = self.get_demand(self.init_inventory_level)
        
        self.data_storing.memorize("time",str(self.data_frame),self.t)
        self.data_storing.memorize("overflow",str(self.data_frame), overlfow/self.t_set)
        self.data_storing.memorize("power",str(self.data_frame), power/self.t_set)
        self.data_storing.memorize("demand",str(self.data_frame), demand/self.t_set)
        
        return self._states


                     
                        
## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class BGLP_RLEnv(Environment):

    C_TYPE = 'MPPS-based BGLP - RL Environment'
    C_CYCLE_LIMIT = 0  # Recommended cycle limit for training episodes


## -------------------------------------------------------------------------------------------------
    def __init__(self,
                 p_reward_type=Reward.C_TYPE_EVERY_AGENT,
                 p_logging=Log.C_LOG_ALL,
                 t_set=10.0,
                 demand=0.1,
                 lr_margin=1.0,
                 lr_demand=4.0,
                 lr_power=0.0010, 
                 margin_p=[0.2,0.8,4],
                 prod_target=10000,
                 prod_scenario='continuous',
                 cycle_limit=0):

        self.num_envs = 5                                                 # Number of internal sub-environments
        self.reward_type = p_reward_type
        
        super().__init__(p_mode = Mode.C_MODE_SIM, 
                         p_latency = None, 
                         p_fct_strans = BGLP4RL(p_name='BGLP_RL', p_logging=p_logging), 
                         p_fct_reward = None, 
                         p_fct_success = None, 
                         p_fct_broken = None, 
                         p_visualize = False, 
                         p_logging = p_logging)
        
        self.C_SCIREF_TYPE    = self.C_SCIREF_TYPE_ARTICLE
        self.C_SCIREF_AUTHOR  = "Dorothea Schwung, Steve Yuwono, Andreas Schwung, Steven X. Ding"
        self.C_SCIREF_TITLE   = "Decentralized learning of energy optimal production policies using PLC-informed reinforcement learning"
        self.C_SCIREF_JOURNAL = "Computers & Chemical Engineering"
        self.C_SCIREF_YEAR    = "2021"
        self.C_SCIREF_MONTH   = "05"
        self.C_SCIREF_DAY     = "28"
        self.C_SCIREF_VOLUME  = "152"
        self.C_SCIREF_DOI     = "10.1016/j.compchemeng.2021.107382"
        
        self.C_CYCLE_LIMIT = cycle_limit
        self.t = 0
        self.t_set = t_set
        self.demand = demand
        self.lr_margin = lr_margin
        self.lr_demand = lr_demand
        self.lr_power = lr_power
        self.prod_target = prod_target
        self.prod_scenario = prod_scenario
        self.margin_p = margin_p
        
        self.data_lists = ["time","overflow","power","demand"]
        self.data_storing = DataStoring(self.data_lists)
        self.data_frame = None
        
        self.set_fill_levels = [0,2,6,8,14,16]
        self.set_overflow = [1,3,7,9,15,17]
        self.set_power = [5,11,13,19,21]
        
        self.reset()


## -------------------------------------------------------------------------------------------------
    @staticmethod
    def setup_spaces():
        state_space = ESpace()
        action_space = ESpace()

        state_space.add_dim(Dimension('R-1 LvlSiloA', 'R', 'Res-1 Level of Silo A', '', '', '', [0, 1]))
        state_space.add_dim(Dimension('R-2 LvlHopperA', 'R', 'Res-2 Level of Hopper A', '', '', '', [0, 1]))
        state_space.add_dim(Dimension('R-3 LvlSiloB', 'R', 'Res-3 Level of Silo B', '', '', '', [0, 1]))
        state_space.add_dim(Dimension('R-4 LvlHopperB', 'R', 'Res-4 Level of Hopper B', '', '', '', [0, 1]))
        state_space.add_dim(Dimension('R-5 LvlSiloC', 'R', 'Res-5 Level of Silo C', '', '', '', [0, 1]))
        state_space.add_dim(Dimension('R-6 LvlHopperC', 'R', 'Res-6 Level of Hopper C', '', '', '', [0, 1]))
        
        action_space.add_dim(Dimension('A-1 Act', 'R', 'Act-1 Belt Conveyor A', '', '', '', [0,1]))
        action_space.add_dim(Dimension('A-2 Act', 'R', 'Act-2 Vacuum Pump B', '', '', '', [0,1]))
        action_space.add_dim(Dimension('A-3 Act', 'Z', 'Act-3 Vibratory Conveyor B', '', '', '', [0,1]))
        action_space.add_dim(Dimension('A-4 Act', 'R', 'Act-4 Vacuum Pump C', '', '', '', [0,1]))
        action_space.add_dim(Dimension('A-5 Act', 'R', 'Act-5 Rotary Feeder C', '', '', '', [0,1]))

        return state_space, action_space



## -------------------------------------------------------------------------------------------------
    def _simulate_reaction(self, p_state: State, p_action: Action) -> State:
        
        # 1. Set values to actuators
        action = []
        for agent_id in p_action.get_agent_ids():
            action_elem = p_action.get_elem(agent_id)
            for action_id in action_elem.get_dim_ids():
                action.append(action_elem.get_value(action_id))
                
        for idx, acts in enumerate(self._fct_strans.get_actuators()-1):
            if idx != len(self._fct_strans.get_actuators())-1:
                boundaries = acts.get_boundaries()
                final_action = action[idx]/(boundaries[1]-boundaries[0])+boundaries[0]
                acts.set_value(final_action)
        
        # 2. Update values of the sensors and component states
        self.init_inventory_level = self._fct_strans.get_component_states()[22].get_value()
        for sig in self._signals:
            if len(sig[1:]) == 1:
                input = sig[1]()
            else:
                input = []
                for x in range(len(sig[1:])):
                    input.append(sig[x+1]())
            sig[0].simulate(input, p_range=self.t_set)

        # 3. Return the resulted states in the form of State object
        self.set_actions(action)
        self._state = self.get_states()
        self._state.set_success(False)
        self._state.set_broken(False)
        
        self.t += self.t_set
        overlfow = self.get_overflow()
        power = self.get_power()
        demand = self.get_demand(self.init_inventory_level)
        
        self.data_storing.memorize("time",str(self.data_frame),self.t)
        self.data_storing.memorize("overflow",str(self.data_frame), overlfow/self.t_set)
        self.data_storing.memorize("power",str(self.data_frame), power/self.t_set)
        self.data_storing.memorize("demand",str(self.data_frame), demand/self.t_set)
        return self._state


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
    def get_margin(self) -> list:
        margin = []
        
        for x in range(len(self.set_fill_levels)):
            fill_level = self._fct_strans.get_component_states()[self.set_fill_levels[x]].get_value()
            boundaries = self._fct_strans.get_component_states()[self.set_fill_levels[x]].get_boundaries()
            norm_fill_level = (fill_level-boundaries[0])/(boundaries[1]-boundaries[0])
            if norm_fill_level < self.margin_p[0]:
                m = (0-self.margin_p[2])/(self.margin_p[0])*(norm_fill_level-self.margin_p[0])*self.t_step
            elif norm_fill_level > self.margin_p[1]:
                m = self.margin_p[2]/(1-self.margin_p[1])*(norm_fill_level-self.margin_p[1])*self.t_step
            else:
                m = 0.0
            margin.append(m)
        return margin


## -------------------------------------------------------------------------------------------------
    def get_overflow(self) -> float:
        total_overflow = 0
        
        for x in range(len(comp_states)):
            total_overflow += self._fct_strans.get_component_states()[self.set_overflow[x]].get_value()
        return total_overflow


## -------------------------------------------------------------------------------------------------
    def get_power(self) -> float:
        total_power = 0
        
        for x in range(len(self.set_power)):
            total_power += self._fct_strans.get_component_states()[self.set_power[x]].get_value()
        return total_power


## -------------------------------------------------------------------------------------------------
    def get_demand(self, init_volume) -> float:
        current_volume = self._fct_strans.get_component_states()[22].get_value()
        delta = current_volume-init_volume
        self.prod_reached += delta
        
        if (self.demand*self.t_set) > delta:
            total_demand = delta-self.demand*self.t_set
        else:
            total_demand = 0
        return total_demand


## -------------------------------------------------------------------------------------------------
    def _compute_reward(self, p_state: State = None, p_state_new: State = None) -> Reward:
        reward = Reward(self.reward_type)

        if self.reward_type == Reward.C_TYPE_OVERALL:
            r_overall = 0
            r_overall = r_overall + sum(self.calc_reward()).item()
            reward.set_overall_reward(r_overall)
        
        elif self.reward_type == Reward.C_TYPE_EVERY_AGENT:
           for agent_id in self._last_action.get_agent_ids():
               r_reward = self.calc_reward()
               reward.add_agent_reward(agent_id, r_reward[int(agent_id)])
               
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
            if self.prod_reached >= self.prod_target:
                self._state.set_terminal(True)
                return True
            else:
                return False


## -------------------------------------------------------------------------------------------------
    def _compute_broken(self, p_state: State) -> bool:
        return False


## -------------------------------------------------------------------------------------------------
    def _reset(self, p_seed=None) -> None:
        random.seed(p_seed)
        self._fct_strans.get_component_states()[23]._function.prod_target = self.demand
        
        for acts in self._fct_strans.get_actuators():
            acts.deactivate()
        for sens in self._fct_strans.get_sensors():
            sens.deactivate()
            
        for st in range(len(self.set_fill_levels)):
            levels_init = random.uniform(0,1)
            self._fct_strans.get_component_states()[self.set_fill_levels[st]].set_value(levels_init)
        self._fct_strans.get_component_states()[22].set_value(0)
        
        self.t              = 0
        self.prod_reached   = 0
        self._state = self.get_states()
        self._state.set_success(False)
        self._state.set_broken(False)
        
        if self.data_frame == None:
            self.data_frame = 0
        else:
            self.data_frame += 1
        self.data_storing.add_frame(str(self.data_frame))
            

## -------------------------------------------------------------------------------------------------
    def calc_reward(self):
        
        margin = self.get_margin()
        power = self.get_power()/self.t_set
        demand = self.get_demand(self.init_inventory_level)/self.t_set
        
        for actnum, pwr in enumerate(self.set_power):
            try:
                power_max = self.get_component_states[pwr]._function.max_power
            except:
                power_max = self.get_component_states[pwr]._function.power
                
            self.reward[actnum] = 1/(1+self.lr_margin*margin[actnum])+1/(1+self.lr_power*power[actnum]/(power_max/1000.0))
            if actnum == len(self.set_power)-1:
                self.reward[actnum] += 1/(1-self.lr_demand*demand)
            else:
                self.reward[actnum] += 1/(1+self.lr_margin*margin[actnum+1])
        return self.reward[:]





## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class MyBGLP(RLScenario):

    C_NAME = 'My_BGLP'
    

## -------------------------------------------------------------------------------------------------
    def _setup(self, p_mode, p_ada, p_visualize, p_logging):
        self._env = BGLP_RLEnv(p_logging=True)
        self._agent = MultiAgent(p_name='Random Policy', p_ada=1, p_logging=False)
        state_space = self._env.get_state_space()
        action_space = self._env.get_action_space()
        
        
        # Agent 1
        _name         = 'BELT_CONVEYOR_A'
        _id           = 0
        _ospace       = state_space.spawn([state_space.get_dim_ids()[0],state_space.get_dim_ids()[1]])
        _aspace       = action_space.spawn([action_space.get_dim_ids()[0]])
        _policy       = RandomGenerator(p_observation_space=_ospace, p_action_space=_aspace, p_buffer_size=1, p_ada=1, p_logging=False)
        self._agent.add_agent(
            p_agent=Agent(
                p_policy=_policy,
                p_envmodel=None,
                p_name=_name,
                p_id=_id,
                p_ada=True,
                p_logging=True),
            p_weight=1.0
            )
        
        
        # Agent 2
        _name         = 'VACUUM_PUMP_B'
        _id           = 1
        _ospace       = state_space.spawn([state_space.get_dim_ids()[1],state_space.get_dim_ids()[2]])
        _aspace       = action_space.spawn([action_space.get_dim_ids()[1]])
        _policy       = RandomGenerator(p_observation_space=_ospace, p_action_space=_aspace, p_buffer_size=1, p_ada=1, p_logging=False)
        self._agent.add_agent(
            p_agent=Agent(
                p_policy=_policy,
                p_envmodel=None,
                p_name=_name,
                p_id=_id,
                p_ada=True,
                p_logging=True),
            p_weight=1.0
            )
        
        
        # Agent 3
        _name         = 'VIBRATORY_CONVEYOR_B'
        _id           = 2
        _ospace       = state_space.spawn([state_space.get_dim_ids()[2],state_space.get_dim_ids()[3]])
        _aspace       = action_space.spawn([action_space.get_dim_ids()[2]])
        _policy       = RandomGenerator(p_observation_space=_ospace, p_action_space=_aspace, p_buffer_size=1, p_ada=1, p_logging=False)
        self._agent.add_agent(
            p_agent=Agent(
                p_policy=_policy,
                p_envmodel=None,
                p_name=_name,
                p_id=_id,
                p_ada=True,
                p_logging=True),
            p_weight=1.0
            )
        
        
        # Agent 4
        _name         = 'VACUUM_PUMP_C'
        _id           = 3
        _ospace       = state_space.spawn([state_space.get_dim_ids()[3],state_space.get_dim_ids()[4]])
        _aspace       = action_space.spawn([action_space.get_dim_ids()[3]])
        _policy       = RandomGenerator(p_observation_space=_ospace, p_action_space=_aspace, p_buffer_size=1, p_ada=1, p_logging=False)
        self._agent.add_agent(
            p_agent=Agent(
                p_policy=_policy,
                p_envmodel=None,
                p_name=_name,
                p_id=_id,
                p_ada=True,
                p_logging=True),
            p_weight=1.0
            )
        
        
        # Agent 5
        _name         = 'ROTARY_FEEDER_C'
        _id           = 4
        _ospace       = state_space.spawn([state_space.get_dim_ids()[4],state_space.get_dim_ids()[5]])
        _aspace       = action_space.spawn([action_space.get_dim_ids()[4]])
        _policy       = RandomGenerator(p_observation_space=_ospace, p_action_space=_aspace, p_buffer_size=1, p_ada=1, p_logging=False)
        self._agent.add_agent(
            p_agent=Agent(
                p_policy=_policy,
                p_envmodel=None,
                p_name=_name,
                p_id=_id,
                p_ada=True,
                p_logging=True),
            p_weight=1.0
            )
        
        return self._agent





## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
logging         = Log.C_LOG_ALL
visualize       = False
dest_path       = str(Path.home())
cycle_limit     = 20000
cycle_per_ep    = 100
eval_freq       = 10
eval_grp_size   = 5
adapt_limit     = 0
stagnant_limit  = 0
score_ma_hor    = 5

training        = RLTraining(
    p_scenario_cls=MyBGLP,
    p_cycle_limit=cycle_limit,
    p_cycles_per_epi_limit=cycle_per_ep,
    p_eval_frequency=eval_freq,
    p_eval_grp_size=eval_grp_size,
    p_adaptation_limit=adapt_limit,
    p_stagnation_limit=stagnant_limit,
    p_score_ma_horizon=score_ma_hor,
    p_collect_states=True,
    p_collect_actions=True,
    p_collect_rewards=True,
    p_collect_training=True,
    p_visualize=visualize,
    p_path=dest_path,
    p_logging=logging
)

training.run()

