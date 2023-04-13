## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - A Synoptic Framework for Standardized Machine Learning Tasks
## -- Package : mlpro_mpps.examples
## -- Module  : howto_006_run_EA_on_LS_using_MPPS.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2023-03-21  0.0.0     ML       Creation
## -- 2023-03-28  1.0.0     ML/SY    Release of first version
## -- 2023-04-13  1.0.1     SY/ML    Code Cleaning and Debugging
## -------------------------------------------------------------------------------------------------

"""
Ver. 1.0.1 (2023-04-13)

This example demonstrates the implementation of the MPPS-based Liquid Laboratroy Station as an EA Environment.

You will learn:
    
    1) How to set up a built-in MPPS to a EA environment.
    
    2) How to set up EA scenario and training, including agent, policies, etc.
    
"""


from mlpro_mpps.pool.ml.ea_environment.EA001_LS import LS_EAEnv
from mlpro.bf.math import *
from mlpro.rl.models import *
from pathlib import Path





## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
# 1 Implement EA agent policy
class EAPolicy(Policy):

    C_NAME      = 'EAPolicy'


## -------------------------------------------------------------------------------------------------
    def __init__(self, 
                 p_observation_space: MSpace, 
                 p_action_space: MSpace, 
                 p_buffer_size=1, 
                 p_ada=True, 
                 p_visualize: bool = False, 
                 p_logging=Log.C_LOG_ALL
                 ):
        super().__init__(p_observation_space, 
                         p_action_space, 
                         p_buffer_size, 
                         p_ada, 
                         p_visualize, 
                         p_logging
                         )
        
        # init population
        self._size = p_buffer_size
        self._population = BufferRnd(p_buffer_size)


## -------------------------------------------------------------------------------------------------
    def compute_action(self) -> Action:

        # create here a way to choose the right action
        action_values = self.individual['action'][0].get_sorted_values()

        return Action(self._id, self._action_space, action_values)  


## -------------------------------------------------------------------------------------------------
    def _store_behaviour(self, p_action:Action, p_state:State, p_reward:Reward):

        # create individual as buffer element
        new_individual = BufferElement(dict(action=p_action,
                                            state=p_state,
                                            reward=p_reward,
                                            reward_value=p_reward.get_agent_reward(self._id)
                                            )
                                      )
        # store buffer element
        self._population.add_element(new_individual)  


## -------------------------------------------------------------------------------------------------
    def _adapt(self) -> bool:
        
        self.ada_buffer = BufferRnd(2)     # adaption buffer
        self.individual = None          # reset individual

        # start evolution process
        self.selection()
        self.recombination()
        self.mutation()
        
        return True
    
    
## -------------------------------------------------------------------------------------------------
    def selection(self):
        """
        This is an minimum example realisation of a selection function and can be changed depending 
        on the approach. 
        """

        
        # select last individual
        self.ada_buffer.add_element(BufferElement(self._population.get_latest()))

        # select random individual
        individual_values = self._population.get_sample(1)
        individual_list = [item for sublist in individual_values.values() for item in sublist]
    
        # select last individual
        self.ada_buffer.add_element(BufferElement(dict(action=individual_list[0],
                                                       state=individual_list[1],
                                                       reward=individual_list[2],
                                                       reward_value=individual_list[2].get_agent_reward(self._id)
                                                       )
                                                  )
                                   )
        
        
## -------------------------------------------------------------------------------------------------
    def recombination(self):
        """
        This is an minimum example realisation of a recombination function and can be changed depending 
        on the approach. 
        """
        # get data from buffer
        all_data = self.ada_buffer.get_all()
        _action = all_data['action']

        # stack action 
        action_values = np.stack((_action[0].get_sorted_values(), 
                                  _action[1].get_sorted_values()
                                  )
                                )

        # recombine 
        action_values = np.mean(action_values, axis=0)

        # write new action to next individual
        self.individual = dict(action=[Action(self._id, self._action_space, action_values)])


## -------------------------------------------------------------------------------------------------
    def mutation(self):
        """
        This is an minimum example mutation of a recombination function and can be changed depending 
        on the approach. 
        """
        # create mask
        mask = np.random.rand(3)                    # init mask
        mask = np.where(mask<0.5, True, False)      # create mask

        # get current values and last values
        action_values = self._population.get_latest()['action'].get_sorted_values()
        individual_values = self.individual['action'][0].get_sorted_values()

        # mutate individual values
        individual_values = np.where(mask==True, action_values, individual_values)
        
        # write new action to next individual
        self.individual = dict(action=[Action(self._id, self._action_space, action_values)])

        



## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
# 2 Implement EA scenario
class EASenario(Scenario):
    """
    This class creates an EA senario where an EA Agent interact with an MPPS environment.
    """
    C_NAME      = 'EASenario - Liquid Laboratory Station'

    def _setup(self, p_mode, p_ada: bool, p_visualize: bool, p_logging) -> Model:

        # 1.1 Setup Environment
        self._env = LS_EAEnv(p_logging=p_logging)
        
        # 2.2 Setup and return standard single-agent with own policy
        return Agent(p_policy=EAPolicy(p_observation_space=self._env.get_state_space(),
                                       p_action_space=self._env.get_action_space(),
                                       p_buffer_size=10,
                                       p_ada=p_ada,
                                       p_visualize=p_visualize,
                                       p_logging=p_logging
                                       ),    
                     p_envmodel=None,
                     p_name='Smith',
                     p_ada=p_ada,
                     p_visualize=p_visualize,
                     p_logging=p_logging
                     )
    

## -------------------------------------------------------------------------------------------------
    def get_latency(self) -> timedelta:
        return self._env.get_latency()
    

## -------------------------------------------------------------------------------------------------
    def _init_population(self):

        
        for individual_num in range(self._model._policy._size):
            action_values = np.ones(self._env.get_action_space().get_num_dim())*random.random()
            action = Action(self._model._policy.get_id(), self._env._action_space, action_values)

            self._env.process_action(action)

            state = self._env.get_state()
            reward = self._env.compute_reward()

            self._model._policy._store_behaviour(action, state, reward)

        # select start individual
        self._model._policy.individual = self._model._policy._population.get_sample(random.randint(1, self._model._policy._size-1))


## -------------------------------------------------------------------------------------------------
    def _run_cycle(self):

        # 1 Agent: compute and log next action
        self.log(self.C_LOG_TYPE_I, 'Process time', self._timer.get_time(), ': Agent computes action...')
        action = self._model._policy.compute_action()
        ts = self._timer.get_time()
        action.set_tstamp(ts)
        
        # 2 Environment: process agent's action
        self.log(self.C_LOG_TYPE_I, 'Process time', self._timer.get_time(), ': Env processes action...')
        self._env.process_action(action)
        self._timer.add_time(self._env.get_latency())
        self._env.get_state().set_tstamp(self._timer.get_time())

        # 3 Environment: get new state
        state = self._env.get_state()
        state.set_tstamp(self._timer.get_time())

        # 4 Environment: compute and log reward
        reward = self._env.compute_reward()
        ts = self._timer.get_time()
        reward.set_tstamp(ts)

        # 5. Agent: store action and behaviour
        self.log(self.C_LOG_TYPE_I, 'Process time', self._timer.get_time(), ': Agent store action and behaviour...')
        self._model._policy._store_behaviour(action, state, reward)

        # 6 Agent: adapt policy
        self.log(self.C_LOG_TYPE_I, 'Process time', self._timer.get_time(), ': Agent adapts policy...')
        adapted = self._model._policy._adapt()

        return False, False, adapted, False



    
    
## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class EATraining(Training):
    """
    This class performs an episodic EA training for a single agent in a given environment. 
    """

    C_NAME      = 'EATraining - Liquid Laboratory Station'

    def __init__(self, **p_kwargs):
        super().__init__(**p_kwargs)

        # Optional parameter p_cycles_per_epi_limit
        try:
            self._cycles_per_epi_limit = self._kwargs['p_cycles_per_epi_limit']
        except KeyError:
            self._cycles_per_epi_limit = -1

        self._mode = self.C_MODE_TRAIN
        self._cycles_episode = 0
        self._num_episodes = 0
        self._num_cycles = 0

        self._scenario._init_population()


## -------------------------------------------------------------------------------------------------
    def _run_cycle(self) -> bool:
        """
        Single custom trainig cycle to be redefined. Custom training results can be added to using
        self._results.add_custom_result(p_name, p_value).
        Returns
        -------
        bool
            True, if training has finished. False otherwise.
        """

        # 0 Intro
        eof_training = False

        # 1 Init next episode
        if self._cycles_episode == 0:

            # reset senario
            self._scenario.reset()

            # log training
            self.log(self.C_LOG_TYPE_W, Training.C_LOG_SEPARATOR)
            self.log(self.C_LOG_TYPE_W, '-- Training episode', self._num_episodes, 'started...')
            self.log(self.C_LOG_TYPE_W, Training.C_LOG_SEPARATOR, '\n')


        # 2 Run a cycle
        success, error, timeout, limit, adapted, end_of_data = self._scenario.run_cycle()
        self._cycles_episode += 1

        if adapted:
            self._results.num_adaptations += 1

        # 3 Update current evaluation
        if self._mode == self.C_MODE_EVAL:

            self._eval_num_cycles += 1
            if limit:
                self._eval_num_limit += 1
            if success:
                self._eval_num_success += 1
            if error:
                self._eval_num_broken += 1

        # 4 Check: Episode finished?
        if self._cycles_episode == self._cycles_per_epi_limit:
            # 4.5 Cycle limit of training setup reached
            self.log(self.C_LOG_TYPE_W, 'Limit of', self._cycles_per_epi_limit, 'cycles per episode reached (Training)')

            # make closing logs
            self.log(self.C_LOG_TYPE_W, Training.C_LOG_SEPARATOR)
            self.log(self.C_LOG_TYPE_W, '-- Training episode', self._num_episodes, 'finished after', str(self._cycles_episode), 'cycles')
            self.log(self.C_LOG_TYPE_W, '-- Training cycles finished:', self._num_cycles + 1)
            self.log(self.C_LOG_TYPE_W, Training.C_LOG_SEPARATOR, '\n\n')

            # increase num_episodes
            self._num_episodes += 1

            # reset cycle episodes
            self._cycles_episode = 0


        # 5 Check: Training finished?
        if (self._adaptation_limit > 0) and (self._results.num_adaptations == self._adaptation_limit):
            self.log(self.C_LOG_TYPE_W, 'Adaptation limit ', str(self._adaptation_limit), ' reached')
            eof_training = True

        # 6 Outro
        return eof_training





# 3 Create scenario and start training
## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    # 3.1 Parameters for demo mode
    cycle_limit = 500          
    epi_limit   = 20
    logging     = Log.C_LOG_WE
    visualize   = None
    path        = None
 
else:
    # 3.2 Parameters for internal unit test
    cycle_limit = 50
    epi_limit   = 10
    logging     = Log.C_LOG_NOTHING
    visualize   = False
    path        = None


# 3.3 Create and run training object
training = EATraining(p_scenario_cls=EASenario,             # init senario
                      p_cycle_limit=cycle_limit,            # set training cyle limit
                      p_cycles_per_epi_limit=epi_limit,     # set episodes per cycle
                      p_path=path,                          # get path to store solutions
                      p_visualize=visualize,                # visualize plots
                      p_logging=logging                     # log trianing process
                      )

# start a run
training.run()