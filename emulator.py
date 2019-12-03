'''
File name: emulator.py
    AI environment emulator class implementation (Implements an AI emulator based
    on Open AI Gym [1])
           
Author: Vasileios Saveris
enail: vsaveris@gmail.com

License: MIT

Date last modified: 01.12.2019

References:
    [1] arXiv:1606.01540 [cs.LG]

Python Version: 3.6
'''

from collections import deque
import numpy as np 
import pandas as pd
import gym

'''
Constants
'''
C_VERBOSE_NONE   = 0  # Printing is disabled
C_VERBOSE_INFO   = 1  # Only information printouts (constructor)
C_VERBOSE_DEBUG  = 2  # Debugging printing level (all printouts)

    
class Emulator():
    ''' 
    Summary:
        Emulator class implementation using OpenAI gym API.

    Private Attributes:
        __verbose: int
            Verbose level (0: None, 1: INFO, 2: DEBUG
        
        __rendering: boolean
            If True, rendering of the OpenAI gym environment is enabled.
            
        __statistics: boolean
            If True, printing and saving of statistics for each episode are both enabled.
        
        __current_state: object
            The current state of the OpenAI gym environment.
            
        __average_reward_episodes: int
            The number of last concecutive episodes in which the average reward should be 
            calculated.
            
        __environment: OpenAI gym environment object
            The environment returned by the make() OpenAI gym function.
            
        __last_x_rewards: deque
            The last X concecutive rewards, where X = __average_reward_episodes.

    Public Attributes:
        emulator_started: boolean
            Indicates the state of the Emulator (True means an episode is running)
        
        average_reward: float
            The average of last X concecutive episode rewards.
            
        episode_number: int
            The sequential number of the episode.
            
        episode_total_reward: float
            The total reward of the current episode.
            
        episode_total_steps: int
            The total steps of the current episode.
            
        state_size: int
            The dimensions of the state object for the given environment.
            
        actions_number: int
            The number of different actions supported by the given environment.
        
        execution_statistics: pandas.DataFrame
            Emulator statistics ('episode', 'steps', 'total_reward', 'last_X_average_reward')

    Private Methods:
        __init__(): returns None
            Class constructor.
        
    Public Methods:
        start(): returns environment's initial state.
            Resets the environment and initializes all the attributes for the current episode.
            Should be used as the first call when we want to start a new episode.
            
        applyAction(action: int or None): returns an experience list [s, a, r, s']
            Apply an action to the environment. If action is none, a random action is selected.
    
    '''

    def __init__(self, scenario, average_reward_episodes = 1, statistics = False, rendering = False, 
                 seed = None, verbose = C_VERBOSE_NONE):
        '''
        summary: 
            Class constructor. Creates the OpenAI gym environment and initializes
            all the attributes.
    
        Args:
            scenario: string
                The OpenAI gym scenario to be emulated.
                
            average_reward_episodes: int
                On how many concecutive episodes the averaged reward should be calculated.

            statistics: boolean
                If True, execution statistics are printed and stored in a class attribute. Verbose can 
                be C_VERBOSE_NONE.
            
            rendering: boolean
                If True, then rendering is enabled and a preview of the environment
                is shown for each state change.
                
            seed: int
                Optional Seed to be used with the OpenAI gym environment, for results reproducability.
                         
            verbose: int
                Verbose level (0: None, 1: INFO, 2: DEBUG, see CONSTANTS section).
                
        Raises:
            -
        
        Returns:
            -
            
        notes:
            -
     
        '''

        self.__verbose = verbose  
        
        if self.__verbose > C_VERBOSE_NONE:
            print('\nRL Emulator initialization (scenario = ', scenario, ', average_reward_episodes = ', average_reward_episodes,
                  ', statistics = ', statistics, ', rendering = ', rendering, ', seed = ', seed, ')', sep = '')
        
        self.__rendering = rendering
        self.__statistics = statistics
        self.__current_state = None
        self.__average_reward_episodes = average_reward_episodes
        
        #Create Open AI gym environment
        try:
            self.__environment = gym.make(scenario)
        except:
            print('ERROR: class Emulator, \'', scenario, '\' is not a valid Open AI Gym scenario, script exits.', sep = '')
            exit(-1) 
            
        #Seed the environment
        if seed is not None:
            self.__environment.seed(seed)

        #Keep track of the last x consecutive total rewards
        self.__last_x_rewards = deque(maxlen = self.__average_reward_episodes)
             
        #Public Attributes
        self.emulator_started = False  #If False the Emulator not started, or episode finished
        self.average_reward = 0
        self.episode_number = 0
        self.episode_total_reward = 0
        self.episode_total_steps  = 0
        
        #Keeps statistics for all the episodes
        if self.__statistics:
            self.execution_statistics = pd.DataFrame(data = None, index = None, columns = ['episode', 'steps', 'total_reward', 
                'last_X_average_reward'], dtype = None, copy = False)
        
        #Get the observation space size based on its type
        if isinstance(self.__environment.observation_space, gym.spaces.box.Box):       
            self.state_size = 1
            for i in range(len(self.__environment.observation_space.shape)):
                self.state_size *= self.__environment.observation_space.shape[i]      
        
        elif isinstance(self.__environment.observation_space, gym.spaces.discrete.Discrete):
            self.state_size = self.__environment.observation_space.n           
        
        else:
            print('ERROR: class RLEmulator, \'', type(self.__environment.observation_space), '\' Observation Space type is not supported, script exits.')
            exit(-1)
            
        #Get the action space size based on its type
        if isinstance(self.__environment.action_space, gym.spaces.box.Box):          
            self.actions_number = 1
            for i in range(len(self.__environment.action_space.shape)):
                self.actions_number *= self.__environment.action_space.shape[i]
       
        elif isinstance(self.__environment.action_space, gym.spaces.discrete.Discrete):
            self.actions_number = self.__environment.action_space.n
            
        else:
            print('ERROR: class RLEmulator, \'', type(self.__environment.observation_space), '\' Actions Space type is not supported, script exits.')
            exit(-1)
            
        if self.__verbose > C_VERBOSE_NONE:
            print('- RL Emulator created (observations_size = ', self.state_size, ', actions_size = ', self.actions_number, ')', sep = '')
        
 
    def start(self):
        '''
        summary: 
            Starts or restarts the Emulator (starts an episode).
    
        Args:
            -
            
        Raises:
            -

        Returns:
            current_state: Environment's state object
                The current (initial) state of the Open AI gym environment. Returned
                after the OpenAI gym reset() function called.
            
        notes:
            -
     
        '''
        
        #Initialze attributes at the begining of the episode.
        self.emulator_started = True       
        self.episode_number += 1
        self.episode_total_reward = 0
        self.episode_total_steps  = 0
        
        #Reset the environment
        self.__current_state = self.__environment.reset()

        if self.__rendering:
            self.__environment.render()
            
        if self.__verbose > C_VERBOSE_INFO:
            print('RL Emulator Started')
  
        return self.__current_state
        
            
    def applyAction(self, action = None):
        '''
        summary: 
            Applies a specific or random action to the environment.
    
        Args:
            action: integer or None
                Action to be applied in the envoronment. If None, then a random action
                should be used.
        
        Raises:
            -
            
        Returns:
            experience_tuple: list
                An experience tuple for this step in the form of [s, a, r, s']
            
        notes:
            -
     
        '''
        
        if not self.emulator_started:
            print('ERROR: Emulator is not started yet, script exits.')
            exit(-1)
            
        #If no action given, select a random one
        if action is None:
            action = self.__environment.action_space.sample()

        new_state, reward, done, info = self.__environment.step(action)
        
        self.episode_total_reward += reward
        self.episode_total_steps  += 1
        
        if self.__rendering:
            self.__environment.render()
        
        if self.__verbose > C_VERBOSE_INFO:
            print('RL Emulator Apply Action (action = ', action, ', reward = ', reward, 
                  ', episode_is_done = ', done, ')', sep = '')
                  
        current_state = self.__current_state #Keep the current state for the return statememt
        
        #Episode finished
        if done:
            new_state = None
        
            #Update Emulator status, the caller has to restart the Emulator for another episode (start())
            self.emulator_started = False
            
            #Calculate average reward
            self.__last_x_rewards.append(self.episode_total_reward)
            self.average_reward = np.mean(self.__last_x_rewards)
            
            if self.__statistics:
                print('Statistics: episode = %05d' % self.episode_number, ', steps = %4d' % self.episode_total_steps, 
                      ', total_reward = %9.3f' % round(self.episode_total_reward, 3), ', ', self.__average_reward_episodes, 
                      '_episodes_average_reward = %9.3f' % round(self.average_reward, 3), sep = '')
                
                #Store the statistics for this episode
                self.execution_statistics.loc[len(self.execution_statistics.index)] = ([self.episode_number, self.episode_total_steps, 
                    round(self.episode_total_reward, 3), round(self.average_reward, 3)])
        
        #Episode is not done yet
        else:
            self.__current_state = new_state     #Update the current state for the next step
        
        #Return an experience tuple (list)
        return [current_state, action, reward, new_state]
        