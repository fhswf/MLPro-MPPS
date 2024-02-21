## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - A Synoptic Framework for Standardized Machine Learning Tasks
## -- Package : mlpro_mpps.examples
## -- Module  : howto_007_run_using_LS_MPPS_data_generation.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2023-03-21  0.0.0     ML       Creation
## -- 2023-03-28  1.0.0     ML/SY    Release of first version
## -- 2023-04-13  1.0.1     SY       Code Cleaning and Debugging
## -------------------------------------------------------------------------------------------------

"""
Ver. 1.0.1 (2023-04-13)

This example demonstrates the implementation of the MPPS-based Liquid Laboratroy Station for data storing.
Based on this a dataset can be generated automatically from a MPPS environment and can be used to train a 
supervised learner or an unsupervised learner.

You will learn:
    
    1) How to set up a built-in MPPS environment.
    
    2) How to set up a scenario and training to create automatically a dataset from MPPS
    
"""


from mlpro_mpps.pool.ml.sl_environment.SL001_LS import LS_SLEnv
from mlpro.bf.math import *
from mlpro.rl.models import *
from pathlib import Path
import torch
from torch.utils.data import Dataset
import pandas as pd





## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------

class CreateCustomDataset(Dataset):
    """
    This class is an example of an custome dataset to predict a timeseries.
    This class is initalised with the path of data loading, target channel generation and the sequence length.
    """


## -------------------------------------------------------------------------------------------------
    def __init__(self, p_path:str, p_channels:list=['Transport', 'Overflow', 'Energy'], p_seq_len=1):
        
        super().__init__()

        # load logged data
        self._df_state = pd.read_csv(os.path.join(p_path+os.sep+'env_states.csv'), sep='\t')
        self._df_action = pd.read_csv(os.path.join(p_path+os.sep+'agent_actions.csv'), sep='\t')
        self._df_reward = pd.read_csv(os.path.join(p_path+os.sep+'env_rewards.csv'), sep='\t')

        # define seq_len
        self._seq_len = p_seq_len

        # create input tensor
        input_data = pd.concat([self._df_state['state'], self._df_action['A-1 Act'], self._df_action['A-2 Act'], self._df_action['A-3 Act']], axis=1)
        self._input_tensor = torch.tensor(input_data.values)
    
        # create target tensor
        self._target_tensor = torch.tensor(self._df_reward[p_channels].values)


## -------------------------------------------------------------------------------------------------
    def __len__(self):
        
        return self._input_tensor.__len__() - 2*self._seq_len


## -------------------------------------------------------------------------------------------------
    def __getitem__(self, index):
        
        return (self._input_tensor[index:index+self._seq_len], self._target_tensor[index+self._seq_len:index+2*self._seq_len])





## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------

class ThisDataStoring(DataStoring):
    
    # Frame ID renamed
    C_VAR0          = 'Episode ID'

    # Variables for episodic detail data storage
    C_VAR_CYCLE     = 'Cycle'
    C_VAR_DAY       = 'Day'
    C_VAR_SEC       = 'Second'
    C_VAR_MICROSEC  = 'Microsecond'


## -------------------------------------------------------------------------------------------------
    def __init__(self, p_space: Set = None):
        
        self.space = p_space

        # Initialization as an episodic detail data storage
        self.variables = [self.C_VAR_CYCLE, self.C_VAR_DAY, self.C_VAR_SEC, self.C_VAR_MICROSEC]
        self.var_space = []

        for dim_id in self.space.get_dim_ids():
            dim = self.space.get_dim(dim_id)
            self.var_space.append(dim.get_name_short())

        self.variables.extend(self.var_space)

        super().__init__(self.variables)


## -------------------------------------------------------------------------------------------------
    def add_episode(self, p_episode_id):
        
        self.add_frame(p_episode_id)
        self.current_episode = p_episode_id


## -------------------------------------------------------------------------------------------------
    def memorize_row(self, p_cycle_id, p_tstamp: timedelta, p_data):
        """
        Memorizes an episodic data row.
        Parameters: 
            p_cycle_id          Cycle id
            p_tstamp            Time stamp
            p_data              Data that meet the dimensionality of the related space
        """

        self.memorize(self.C_VAR_CYCLE, self.current_episode, p_cycle_id)
        self.memorize(self.C_VAR_DAY, self.current_episode, p_tstamp.days)
        self.memorize(self.C_VAR_SEC, self.current_episode, p_tstamp.seconds)
        self.memorize(self.C_VAR_MICROSEC, self.current_episode, p_tstamp.microseconds)

        for i, var in enumerate(self.var_space):
            self.memorize(var, self.current_episode, p_data[i])





## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------

class RandomActionGenerator(Policy):

    C_NAME      = 'RandomActionGenerator'
       
    
## -------------------------------------------------------------------------------------------------
    def compute_action(self) -> Action:

        my_action_values = np.zeros(self._action_space.get_num_dim())

        for d in range(self._action_space.get_num_dim()):
            my_action_values[d] = random.random() 

        return Action(self._id, self._action_space, my_action_values)
    




## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
# Implement DataGenerator
class DataGenerator(Scenario):
    """
    This class creates an DataGenerator what interacts with the LS MPPS environment.
    """
    C_NAME      = 'DataGenerator - Liquid Laboratory Station'


## -------------------------------------------------------------------------------------------------
    def __init__(self,
                 p_mode=Mode.C_MODE_SIM,  
                 p_ada:bool=True,  
                 p_cycle_limit=0, 
                 p_visualize:bool=True,  
                 p_logging=Log.C_LOG_ALL):  

        # Setup entire scenario
        self._env : Environment = None

        super().__init__(p_mode=p_mode, 
                         p_ada=p_ada, 
                         p_cycle_limit=p_cycle_limit, 
                         p_visualize=p_visualize,
                         p_logging=p_logging
                         )
        
        if self._env is None:
            raise ImplementationError('Please bind your environment to self._env')
        
        # data logging
        self._ds_states = None
        self._ds_actions = None
        self._ds_rewards = None


## -------------------------------------------------------------------------------------------------
    def _setup(self, p_mode, p_ada: bool, p_visualize: bool, p_logging) -> Model:

        # Setup Environment
        """
        As example is the Liquid Laboratory Station as EA environment selected.
        This approach can be expanded to other environments.
        """
        self._env = LS_SLEnv(p_logging=p_logging)
        
        # Setup and return standard single-agent with own policy
        return Agent(p_policy=RandomActionGenerator(p_observation_space=self._env.get_state_space(),
                                                    p_action_space=self._env.get_action_space(),
                                                    p_buffer_size=10,
                                                    p_ada=p_ada,
                                                    p_visualize=p_visualize,
                                                    p_logging=p_logging
                                                    ),    
                     p_envmodel=None,
                     p_name='DerArbeiter',
                     p_ada=p_ada,
                     p_visualize=p_visualize,
                     p_logging=p_logging
                     )
    
    
## -------------------------------------------------------------------------------------------------
    def get_latency(self) -> timedelta:
        
        return self._env.get_latency()


## -------------------------------------------------------------------------------------------------
    def _run_cycle(self):

        # 0 Environment: get current state
        state = self._env.get_state()
        state.set_tstamp(self._timer.get_time())

        # 1 Agent: compute and log action
        self.log(self.C_LOG_TYPE_I, 'Process time', self._timer.get_time(), ': Agent computes action...')
        action = self._model._policy.compute_action()
        ts = self._timer.get_time()
        action.set_tstamp(ts)
        self._ds_actions.memorize_row(self._cycle_id, ts, action.get_sorted_values())
        
        # 2 Environment: process agent's action
        self.log(self.C_LOG_TYPE_I, 'Process time', self._timer.get_time(), ': Env processes action...')
        self._env.process_action(action)
        self._timer.add_time(self._env.get_latency())
        self._env.get_state().set_tstamp(self._timer.get_time())

        # 3 Environment: get new state
        new_state = self._env.get_state()
        new_state.set_tstamp(self._timer.get_time())
        self._ds_states.memorize_row(self._cycle_id, self._timer.get_time(), [state.get_values()[0], new_state.get_values()[0]])

        # 4 Environment: compute and log reward
        reward = self._env.compute_reward()
        ts = self._timer.get_time()
        reward.set_tstamp(ts)
        self._ds_rewards.memorize_row(self._cycle_id, ts, reward.get_action_reward(reward.agent_ids[0]))

        return False, False, True, False
    
    
## -------------------------------------------------------------------------------------------------
    def get_latency(self) -> timedelta:
        
        return self._env.get_latency()





## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------

# 1 Implement your own agent policy
class MyPolicy (Policy):

    C_NAME      = 'MyPolicy'


## -------------------------------------------------------------------------------------------------
    def set_random_seed(self, p_seed=None):
        random.seed(p_seed)


## -------------------------------------------------------------------------------------------------
    def compute_action(self, p_state: State) -> Action:
        # Create a numpy array for your action values 
        my_action_values = np.zeros(self._action_space.get_num_dim())

        # Computing action values is up to you...
        for d in range(self._action_space.get_num_dim()):
            my_action_values[d] = random.random() 

        # Return an action object with your values
        return Action(self._id, self._action_space, my_action_values)


## -------------------------------------------------------------------------------------------------
    def _adapt(self) -> bool:
        # Adapting neural network with the created dataset
        self.log(self.C_LOG_TYPE_I, 'Sorry, I am a stupid agent...')

        # Only return True if something has been adapted...
        return False





## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------

# Implement your own SL scenario
class SLScenario(Scenario):

    C_NAME      = 'Matrix'


## -------------------------------------------------------------------------------------------------
    def _setup(self, p_mode, p_ada: bool, p_visualize: bool, p_logging) -> Model:
        # Setup Environment
        """
        As example is the Liquid Laboratory Station as EA environment selected.
        This approach can be expanded to other environments.
        """
        self._env = LS_SLEnv(p_logging=p_logging)

        # Setup and return standard single-agent with own policy
        return Agent(p_policy=MyPolicy(p_observation_space=self._env.get_state_space(),
                                       p_action_space=self._env.get_action_space(),
                                       p_buffer_size=10,
                                       p_ada=p_ada,
                                       p_visualize=p_visualize,
                                       p_logging=p_logging
                                       ),    
                     p_envmodel=None,
                     p_name='DerLerner',
                     p_ada=p_ada,
                     p_visualize=p_visualize,
                     p_logging=p_logging
                     )
    
    
## -------------------------------------------------------------------------------------------------
    def get_env(self):
        return self._env


## -------------------------------------------------------------------------------------------------
    def get_latency(self) -> timedelta:
        return self._env.get_latency()
    
    
## -------------------------------------------------------------------------------------------------
    def _run_cycle(self):

        # agent neural network
        self.log(self.C_LOG_TYPE_I, 'Process time', self._timer.get_time(), ': Agent adapts policy...')
        adapted = self._model._policy._adapt()


 
        return False, False, adapted, False




    
    
## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class SupervisedLearner(Training):
    """
    This class performs an SupervisedLearner.
    It contains two senariso:
    1) DataGenerator - Interacts with the LS-MPPS and colleted data 
    2) SupervisedSenario - Learns and interprets the system behaviour on the collected data

    This class shows the interaction of the of machine learning algorithm with simulated custon environments.
    As example we present the interaction with the DataGenerator with a simple SupervisedSenario.
    """

    C_NAME      = 'SupervisedLearner'


## -------------------------------------------------------------------------------------------------
    def __init__(self, **p_kwargs):
        super().__init__(**p_kwargs)

        # Mandatory parameter p_scenario_dg - DataGeneration
        try:
            scenario_dg = self._kwargs['p_scenario_dg']
        except:
            raise ParamError('Mandatory parameter p_scenario_dg not supplied')
        
        # Mandatory parameter p_scenario_cls - SupervisedLearning
        try:
            scenario_cls = self._kwargs['p_scenario_cls']
        except:
            raise ParamError('Mandatory parameter p_scenario_cls not supplied')

        # Optional parameter p_cycles_per_epi_limit
        try:
            self._cycles_per_epi_limit = self._kwargs['p_cycles_per_epi_limit']
        except KeyError:
            self._cycles_per_epi_limit = -1

        # Optional parameter _num_episode_limit
        try:
            num_episode_limit = self._kwargs['p_data_epi_limit']
        except KeyError:
            num_episode_limit = -1

        # Optional environment mode
        try:
            env_mode = self._kwargs['p_env_mode']
        except:
            env_mode = Mode.C_MODE_SIM
            self._kwargs['p_env_mode'] = env_mode

        if self._hpt is None:
            try:
                self._data_scenario = scenario_dg(p_mode=env_mode, 
                                                  p_ada=True,
                                                  p_cycle_limit=self._cycle_limit,
                                                  p_visualize=visualize,
                                                  p_logging=logging
                                                  )
                
            except:
                raise ParamError('Par p_scenario_cls: class "' + scenario_cls.__name__ + '" not compatible')
            
            try:
                self._scenario = scenario_cls(p_mode=env_mode, 
                                              p_ada=True,
                                              p_cycle_limit=self._cycle_limit,
                                              p_visualize=visualize,
                                              p_logging=logging
                                              )
            except:
                raise ParamError('Par p_scenario_cls: class "' + scenario_cls.__name__ + '" not compatible')

        # make sure to have same env
        self._env = self._scenario.get_env()

        # limit and counter for data generation
        self._num_episode_limit = num_episode_limit     # max length of data
        self._num_episodes = 0                          # current cycle

        self._mode = self.C_MODE_TRAIN
        self._cycles_episode = 0
        self._num_episodes = 0
        self._num_cycles = 0


## -------------------------------------------------------------------------------------------------
    def _run(self) -> TrainingResults:

        # create dataset with DataGenerator
        self._data       = self._init_data_results()
        while not self.generate_data(): pass
      
        # reset num episodes
        self._num_episodes = 0

        # start SupervisedLearning
        self._new_run = True
        while not self.run_cycle(): pass
        return self.get_results()

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


        # 2 Run a training cycle
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
        if (self._adaptation_limit > 0):
            self.log(self.C_LOG_TYPE_W, 'Adaptation limit ', str(self._adaptation_limit), ' reached')
            eof_training = True


        # 6 Make testing of trained network 

        # 7 Outro
        return eof_training


## -------------------------------------------------------------------------------------------------
    def _init_data_results(self) -> TrainingResults:
        results = super()._init_results()

        # state
        state_space = Set()
        state_space.add_dim(Dimension('state'))
        state_space.add_dim(Dimension('new_state'))


        # reward
        reward_space = Set()
        reward_space.add_dim(Dimension('Transport'))
        reward_space.add_dim(Dimension('Overflow'))
        reward_space.add_dim(Dimension('Energy'))

        
        # connect data logger
        self._data_scenario._ds_states = ThisDataStoring(state_space)       # state
        self._data_scenario._ds_actions = ThisDataStoring(self._env.get_action_space())     # action
        self._data_scenario._ds_rewards = ThisDataStoring(reward_space)                     # reward

        return results
    
## -------------------------------------------------------------------------------------------------
    def generate_data(self):
        """
        This class is an example to create data from mpps environment.
        This function store the cycles and create a pytorch dataset.
        Returns
        -------
        success : bool
            True on success. False otherwise.
        """

        # connect data logger
        self._data_scenario._ds_states.add_episode(self._num_episodes)
        self._data_scenario._ds_actions.add_episode(self._num_episodes)
        self._data_scenario._ds_rewards.add_episode(self._num_episodes)

        # run data generation cycle
        success, error, timeout, limit, adapted, end_of_data = self._data_scenario.run_cycle()

        # count episodes
        self._num_episodes += 1

        # break condition for data generation
        if self._num_episodes >= self._num_episode_limit:

            # store the logged parameter
            self._data_scenario._ds_states.save_data(self._root_path, 'env_states')
            self._data_scenario._ds_actions.save_data(self._root_path, 'agent_actions')
            self._data_scenario._ds_rewards.save_data(self._root_path, 'env_rewards')

            # create a pytorch datased from the logged data
            self._dataset = CreateCustomDataset(p_path=self._root_path, 
                                                #p_channels=['Transport', 'Overflow'], 
                                                #p_seq_len=3
                                                )

            # leaf data generation loop
            end_of_data = True

        return end_of_data





## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------

# Create scenario and start training
if __name__ == "__main__":
    # Parameters for demo mode
    cycle_limit     = 500          
    epi_limit       = 20
    data_epi_limit  = 2000
    logging         = Log.C_LOG_WE
    visualize       = None
    path            = str(Path.home())


    # Create and run training object
    training = SupervisedLearner(p_scenario_dg=DataGenerator,          # init DataGenerator senario
                                 p_scenario_cls=SLScenario,            # init SuperviseLearning senario
                                 p_cycle_limit=cycle_limit,            # set training cyle limit
                                 p_cycles_per_epi_limit=epi_limit,     # set episodes per cycle
                                 p_data_epi_limit=data_epi_limit,      # length of the created dataset 
                                 p_path=path,                          # get path to store solutions
                                 p_visualize=visualize,                # visualize plots
                                 p_logging=logging                     # log trianing process
                                 )
    
    # start a run
    training.run()