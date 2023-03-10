## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - A Synoptic Framework for Standardized Machine Learning Tasks
## -- Package : mlpro_mpps.examples
## -- Module  : howto_005_run_RL_on_LS_using_MPPS.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2023-03-10  0.0.0     ML       Creation
## -------------------------------------------------------------------------------------------------

"""
Ver. 1.0.4 (2023-02-17)

This example demonstrates the implementation of the MPPS-based BGLP as an RL Environment.

You will learn:
    
    1) How to set up a built-in MPPS to a RL environment.
    
    2) How to set up RL scenario and training, including agents, policies, etc.
    
    3) How to incoporate SB3 algorithm into the RL training
    
"""


from mlpro_mpps.pool.rl_environment.RL002_LS import LS_RLEnv
from mlpro.bf.math import *
from mlpro.rl.models import *
from stable_baselines3 import PPO, A2C
from mlpro.wrappers.sb3 import WrPolicySB32MLPro
from copy import deepcopy
import torch
from pathlib import Path



                        
# 2 Implement your own RL scenario
class RL_Liquid_Station_Scenario(RLScenario):

    C_NAME = 'RL_Liquid_Station_Scenario'

## -------------------------------------------------------------------------------------------------
    def _setup(self, p_mode, p_ada, p_visualize, p_logging):

        # 1.1 Setup Multi-Agent Environment (consisting of 3 OpenAI Gym Cartpole envs)
        self._env = LS_RLEnv(p_logging=p_logging)
        
        # Get state and action space
        state_space = self._env.get_state_space()
        action_space = self._env.get_action_space()
        

        # 1.2 Setup Multi-Agent 

        # 1.2.1 Create empty Multi-Agent
        self._agent = MultiAgent(p_name='SB3 Policy', 
                                 p_ada=1, 
                                 p_logging=p_logging
                                 )
        
        
        
        policy_kwargs = dict(activation_fn=torch.nn.ReLU,
                             net_arch=[dict(pi=[128, 128], vf=[128, 128])]
                             )
        
        policy_sb3 = PPO(policy="MlpPolicy",
                         n_steps=100,
                         env=None,
                         _init_setup_model=False,
                         policy_kwargs=policy_kwargs,
                         device="cpu",
                         seed=2
                         )
        
        # Agent 1 
        _name         = 'Agent - Pump 1'
        _id           = 0
        _ospace       = state_space.spawn([state_space.get_dim_ids()[0],state_space.get_dim_ids()[1]])
        _aspace       = action_space.spawn([action_space.get_dim_ids()[0]])
        
        _policy_wrapped = WrPolicySB32MLPro(p_sb3_policy=deepcopy(policy_sb3),
                                            p_cycle_limit=self._cycle_limit,
                                            p_observation_space=_ospace,
                                            p_action_space=_aspace,
                                            p_ada=p_ada,
                                            p_visualize=p_visualize,
                                            p_logging=p_logging
                                            )
        
        self._agent.add_agent(p_agent=Agent(p_policy=_policy_wrapped,
                                            p_envmodel=None,
                                            p_name=_name,
                                            p_id=_id,
                                            p_ada=True,
                                            p_logging=True),
                                            p_weight=1.0
                                            )
        
        
        # Agent 2
        _name         = 'Agent - Pump 2'
        _id           = 1
        _ospace       = state_space.spawn([state_space.get_dim_ids()[1],state_space.get_dim_ids()[2]])
        _aspace       = action_space.spawn([action_space.get_dim_ids()[1]])
        
        _policy_wrapped = WrPolicySB32MLPro(p_sb3_policy=deepcopy(policy_sb3),
                                            p_cycle_limit=self._cycle_limit,
                                            p_observation_space=_ospace,
                                            p_action_space=_aspace,
                                            p_ada=p_ada,
                                            p_visualize=p_visualize,
                                            p_logging=p_logging
                                            )
        
        self._agent.add_agent(p_agent=Agent(
                              p_policy=_policy_wrapped,
                              p_envmodel=None,
                              p_name=_name,
                              p_id=_id,
                              p_ada=True,
                              p_logging=True),
                              p_weight=1.0
                              )
        
        
        # Agent 3
        _name         = 'Agent - Pump 3'
        _id           = 2
        _ospace       = state_space.spawn([state_space.get_dim_ids()[2],state_space.get_dim_ids()[3]])
        _aspace       = action_space.spawn([action_space.get_dim_ids()[2]])
        
        _policy_wrapped = WrPolicySB32MLPro(p_sb3_policy=deepcopy(policy_sb3),
                                            p_cycle_limit=self._cycle_limit,
                                            p_observation_space=_ospace,
                                            p_action_space=_aspace,
                                            p_ada=p_ada,
                                            p_visualize=p_visualize,
                                            p_logging=p_logging
                                            )
        
        self._agent.add_agent(p_agent=Agent(
                              p_policy=_policy_wrapped,
                              p_envmodel=None,
                              p_name=_name,
                              p_id=_id,
                              p_ada=True,
                              p_logging=True),
                              p_weight=1.0
                              )
        
        # 2.3 Adaptive ML model (here: our multi-agent) is returned
        return self._agent





# 3 Create scenario and start training
if __name__ == "__main__":
    # 3.1 Parameters for demo mode
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
    # 3.2 Parameters for internal unit test
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


# 3.3 Create and run training object
training = RLTraining(p_scenario_cls=RL_Liquid_Station_Scenario,
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