## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - A Synoptic Framework for Standardized Machine Learning Tasks
## -- Package : mlpro_mpps.examples
## -- Module  : howto_003_run_RL_on_BGLP_using_MPPS.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2023-01-03  0.0.0     SY       Creation
## -- 2023-01-16  1.0.0     SY       Release of first version
## -- 2023-02-01  1.0.1     SY       Refactoring
## -- 2023-02-13  1.0.2     SY       Renaming module and refactoring
## -- 2023-02-15  1.0.3     SY       Incorporating SB3 algorithm via SB3 Wrapper of MLPro
## -- 2023-02-17  1.0.4     SY       Optimizing for paper
## -- 2023-02-28  1.0.5     SY       Update following new pool.ml
## -- 2023-11-14  1.0.6     SY       Refactoring
## -------------------------------------------------------------------------------------------------

"""
Ver. 1.0.6 (2023-11-14)

This example demonstrates the implementation of the MPPS-based BGLP as an RL Environment.

You will learn:
    
    1) How to set up a built-in MPPS to a RL environment.
    
    2) How to set up RL scenario and training, including agents, policies, etc.
    
    3) How to incoporate SB3 algorithm into the RL training
    
"""


from mlpro_mpps.pool.ml.rl_environment.RL001_BGLP import BGLP_RLEnv
from mlpro.bf.math import *
from mlpro.rl.models import *
from stable_baselines3 import PPO, A2C
from mlpro.wrappers.sb3 import WrPolicySB32MLPro
from copy import deepcopy
import torch
from pathlib import Path



                     
                        
## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class MyBGLP(RLScenario):

    C_NAME = 'My_BGLP'
    

## -------------------------------------------------------------------------------------------------
    def _setup(self, p_mode, p_ada, p_visualize, p_logging):
        self._env = BGLP_RLEnv(p_logging=p_logging)
        self._agent = MultiAgent(p_name='SB3 Policy', p_ada=1, p_logging=p_logging)
        state_space = self._env.get_state_space()
        action_space = self._env.get_action_space()
        
        policy_kwargs = dict(activation_fn=torch.nn.ReLU,
                              net_arch=[dict(pi=[128, 128], vf=[128, 128])])
        
        policy_sb3 = PPO(
            policy="MlpPolicy",
            n_steps=100,
            env=None,
            _init_setup_model=False,
            policy_kwargs=policy_kwargs,
            device="cpu",
            seed=2)
        
        # Agent 1
        _name         = 'BELT_CONVEYOR_A'
        _id           = 0
        _ospace       = state_space.spawn([state_space.get_dim_ids()[0],state_space.get_dim_ids()[1]])
        _aspace       = action_space.spawn([action_space.get_dim_ids()[0]])
        
        _policy_wrapped = WrPolicySB32MLPro(
            p_sb3_policy=deepcopy(policy_sb3),
            p_cycle_limit=self._cycle_limit,
            p_observation_space=_ospace,
            p_action_space=_aspace,
            p_ada=p_ada,
            p_visualize=p_visualize,
            p_logging=p_logging)
        
        self._agent.add_agent(
            p_agent=Agent(
                p_policy=_policy_wrapped,
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
        
        _policy_wrapped = WrPolicySB32MLPro(
            p_sb3_policy=deepcopy(policy_sb3),
            p_cycle_limit=self._cycle_limit,
            p_observation_space=_ospace,
            p_action_space=_aspace,
            p_ada=p_ada,
            p_visualize=p_visualize,
            p_logging=p_logging)
        
        self._agent.add_agent(
            p_agent=Agent(
                p_policy=_policy_wrapped,
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
        
        _policy_wrapped = WrPolicySB32MLPro(
            p_sb3_policy=deepcopy(policy_sb3),
            p_cycle_limit=self._cycle_limit,
            p_observation_space=_ospace,
            p_action_space=_aspace,
            p_ada=p_ada,
            p_visualize=p_visualize,
            p_logging=p_logging)
        
        self._agent.add_agent(
            p_agent=Agent(
                p_policy=_policy_wrapped,
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
        
        _policy_wrapped = WrPolicySB32MLPro(
            p_sb3_policy=deepcopy(policy_sb3),
            p_cycle_limit=self._cycle_limit,
            p_observation_space=_ospace,
            p_action_space=_aspace,
            p_ada=p_ada,
            p_visualize=p_visualize,
            p_logging=p_logging)
        
        self._agent.add_agent(
            p_agent=Agent(
                p_policy=_policy_wrapped,
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
        
        _policy_wrapped = WrPolicySB32MLPro(
            p_sb3_policy=deepcopy(policy_sb3),
            p_cycle_limit=self._cycle_limit,
            p_observation_space=_ospace,
            p_action_space=_aspace,
            p_ada=p_ada,
            p_visualize=p_visualize,
            p_logging=p_logging)
        
        self._agent.add_agent(
            p_agent=Agent(
                p_policy=_policy_wrapped,
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

if __name__ == "__main__":
    logging         = Log.C_LOG_ALL
    visualize       = False
    dest_path       = str(Path.home())
    cycle_limit     = 200000
    cycle_per_ep    = 1000
    eval_freq       = 0
    eval_grp_size   = 0
    adapt_limit     = 0
    stagnant_limit  = 0
    score_ma_hor    = 0
else:
    logging         = Log.C_LOG_NOTHING
    visualize       = False
    dest_path       = None
    cycle_limit     = 10
    cycle_per_ep    = 10
    eval_freq       = 0
    eval_grp_size   = 0
    adapt_limit     = 0
    stagnant_limit  = 0
    score_ma_hor    = 0
    
training = RLTraining(
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
    p_logging=logging)

training.run()
if __name__ == "__main__":
    training._scenario.get_env().data_storing.save_data(training._root_path, 'bglp')

