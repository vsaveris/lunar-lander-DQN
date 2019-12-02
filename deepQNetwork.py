'''
File name: deepQNetwork.py
    Deep Q-Network class implementation (Implements a DQN [1])
           
Author: Vasileios Saveris
enail: vsaveris@gmail.com

License: MIT

Date last modified: 02.12.2019

References:
    [1] arXiv:1312.5602 [cs.LG]

Python Version: 3.6
'''

import memory as mem
import numpy as np

'''
Constants
'''
C_VERBOSE_NONE   = 0  # Printing is disabled
C_VERBOSE_INFO   = 1  # Only information printouts (constructor)
C_VERBOSE_DEBUG  = 2  # Debugging printing level (all printouts)


class DeepQNetwork(object):
    ''' 
    Summary:
        Deep Q-Network class implementation (Implements a DQN [1])
    
    Private Attributes:
        __dnn: DeepNN object
            The Deep Neural Network to be used
                
        __states_size: int
            The number of elements in the environment's state.
                
        __actions_number: int
            The number of possible actions.
                
        __memory_size: int
            The size of the replay memory feature which will be used by the DQN.
                
        __minibatch_size: int
            The minibatch size which will be retrieved randomly from the memory in each 
            iteration.
                
        __gamma: float
            The discount factor to be used in the equation (3) of [1].
                
        __epsilon: float
            The probability to select a random action (See [1] Algorithm 1).
            
        __epsilon_decay_factor: float
            The decay factor of epsilon parameter, for each iteration step.
                
        __seed: int or None
            Seed to be used with the numpy random generator.
            
        __verbose: int
            Verbose level (0: None, 1: INFO, 2: DEBUG, see CONSTANTS section)
            
    Public Attributes:
        -
    
    Private Methods:
        __init__(emulator, dnn, states_size, actions_number, memory_size, minibatch_size, 
                gamma, epsilon, epsilon_decay_factor, seed, verbose): returns none
            Class constructor.
        
    Public Methods:
        decideAction(state): returns an action (random or optimal).
            Decides the action based on the given state.
    
        storeTransition(experience): returns None
            Adds an experience [s, a, r, s'] in the memory.
    
        sampleRandomMinibatch(): returns None
            Samples a random minibatch from the memory and trains with it the DNN.
        
    '''

    
    def __init__(self, emulator, dnn, states_size, actions_number, memory_size, minibatch_size, gamma, epsilon, 
                 epsilon_decay_factor, seed = None, verbose = C_VERBOSE_NONE):
        '''
        Summary: 
            Class constructor.
    
        Args: 
            dnn: DeepNN object
                The Deep Neural Network to be used
                
            states_size: int
                The number of elements in the environment's state.
                
            actions_number: int
                The number of possible actions.
                
            memory_size: int
                The size of the replay memory feature which will be used by the DQN.
                
            minibatch_size: int
                The minibatch size which will be retrieved randomly from the memory in each 
                iteration.
                
            gamma: float
                The discount factor to be used in the equation (3) of [1].
                
            epsilon: float
                The probability to select a random action (See [1] Algorithm 1).
            
            epsilon_decay_factor: float
                The decay factor of epsilon parameter, for each iteration step.
                
            seed: int or None
                Seed to be used with the numpy random generator.
            
            verbose: int
                Verbose level (0: None, 1: INFO, 2: DEBUG, see CONSTANTS section)
                
        Raises:
            -
        
        Returns:
            -
            
        notes:
            -
     
        '''
        
        self.__verbose = verbose  
            
        if self.__verbose > C_VERBOSE_NONE:
            print('\nDeep Q Network object created (states_size = ', states_size, ', actions_number = ', actions_number,
                  ', memory_size = ', memory_size, ', minibatch_size = ', minibatch_size, ', gamma = ', gamma,
                  ', epsilon = ', epsilon, ', epsilon_decay_factor = ', epsilon_decay_factor, ', seed = ', seed, 
                  ')', sep = '')
        
        #Seed the numpy random number generator
        if seed is not None:
            np.random.seed(seed)
            
        self.__dnn = dnn
        self.__actions_number = actions_number
        self.__states_size = states_size
        
        #Create a memory object instance
        self.__memory = mem.Memory(size = memory_size, type = mem.MemoryType.ROTATE, seed = seed, verbose = self.__verbose)
        
        self.__minibatch_size = minibatch_size
        self.__gamma = gamma
        self.__epsilon = epsilon 
        self.__epsilon_decay_factor = epsilon_decay_factor
        
        
    def decideAction(self, state):
        '''
        Summary: 
            Decides the action based on the given state.
    
        Args: 
            state: state object
                The state from which an action should be decided.
                
        Raises:
            -
        
        Returns:
            action: int
                The selected action (random or optimal).
            
        notes:
            -
     
        '''
            
        #With probability epsilon select a random action
        if np.random.random() < self.__epsilon:
            action = np.random.randint(0, self.__actions_number)
        #Otherwise select the best action from the DNN model
        else:
            action = np.argmax(self.__dnn.predict(state))
            
        if self.__verbose > C_VERBOSE_INFO:
            print('DQN Decide Action (state = ', state, ', action = ', action, ')', sep = '')
            
        #Reduce epsilon based on the decay factor
        self.__epsilon *= self.__epsilon_decay_factor
        
        return action
        
        
    def storeTransition(self, experience):
        '''
        Summary: 
            Adds an experience tuple in the memory.
    
        Args: 
            experience: list
                An experience tuple [s, a, r, s']

        Raises:
            -
            
        Returns:
            -
            
        notes:
            -
     
        '''
        
        if self.__verbose > C_VERBOSE_INFO:
            print('DQN Store Transition (experience = ', experience, ')', sep = '')
                    
        self.__memory.add(experience)
        
        
    def sampleRandomMinibatch(self):
        '''
        Summary: 
            Samples a random minibatch from the memory and trains with it the DNN.
    
        Args: 
            -
                
        Raises:
            -
            
        Returns:
            -
            
        notes:
            -
     
        '''
      
        minibatch = self.__memory.get(self.__minibatch_size)
        
        if self.__verbose > C_VERBOSE_INFO:
            print('DQN Sample Random Minibatch (minibatch_length = ', minibatch.shape[0], 
                  ', minibatch = ', minibatch, ')', sep = '')
        
        #End state s' was stored in memory as None (see class rlEmulator)
        #Replace it with a zero array for passing it to the model
        for i in range(minibatch.shape[0]):
            if minibatch[i, 3] is None:
                minibatch[i, 3] = np.zeros((self.__states_size), dtype = np.float64)
        
        #Predictions for starting state s and end state s' (see Algorithm 1 in [1])
        Q_start_state = self.__dnn.predict(np.stack(minibatch[:, 0], axis = 0))  #Q(s,a)   
        Q_end_state   = self.__dnn.predict(np.stack(minibatch[:, 3], axis = 0))  #Q(s',a)
        
        #Update the Q(s,a) according the Algorithm 1 in [1]
        for i in range(minibatch.shape[0]):
            #End state is a terminal state
            if np.array_equal(minibatch[i, 3], np.zeros((8))):
                Q_start_state[i, minibatch[i, 1]] = minibatch[i, 2]
            else:
                Q_start_state[i, minibatch[i, 1]] = minibatch[i, 2] + self.__gamma*np.amax(Q_end_state[i,:])
                
        #Train the dnn with the updated values of the Q(s,a)
        self.__dnn.train(np.stack(minibatch[:, 0], axis = 0), Q_start_state)
        
        