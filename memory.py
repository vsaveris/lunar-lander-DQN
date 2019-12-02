'''
File name: memory.py
    Memory class implementation (Replay Memory implementation [1])
           
Author: Vasileios Saveris
enail: vsaveris@gmail.com

License: MIT

Date last modified: 02.12.2019

@References:
    [1] arXiv:1312.5602 [cs.LG]

Python Version: 3.6
'''

from collections import deque
import numpy as np

'''
Enumerations
'''
from enum import Enum

class MemoryType(Enum):
    ROTATE = 0  #If maximum len of deque is reached, then oldest entries are discarted
    FILL   = 1  #If maximum len of deque is reached, then no more entries are allowed

'''
Constants
'''
C_VERBOSE_NONE   = 0  # Printing is disabled
C_VERBOSE_INFO   = 1  # Only information printouts (constructor)
C_VERBOSE_DEBUG  = 2  # Debugging printing level (all printouts)


class Memory(object):
    '''    
    Summary:    
        Memory class implementation (Replay Memory implementation [1])    
        
    Private Attributes:    
        __verbose: int    
            Verbose level (0: None, 1: Info, 2: Debug, see CONSTANTS section)    
                
        __size: integer    
            The memory size, in number of objects.    
                
        __type: MemoryType    
            The memory type (see enum MemoryType for supported values).    
                
        __memory: deque    
            The memory, a deque holding the memories (objects of any type).    
        
    Public Attributes:    
        memory_usage: float    
            The usage percentage of the memory (used capacity %). If the memory is not bounded    
            the usage percentage is always 0.0%.    
        
    Private Methods:    
        __init__(): returns none    
            Class constructor.    
                
        __view(): returns None    
            Prints the memory contents. Is called when verbose is set to C_VERBOSE_DEBUG.    
            
    Public Methods:    
        add(memory object): returns None    
            Adds an object in the memory deque.    
                
        get(size int): returns a list of objects    
            Gets the requested number of random memories from the memory deque.    
                
        resizeMemory(new_size int): returns None    
            Resizes the memory to the new_size. The data from the old memory are     
            copied to the new memory. In case the new_size cannot hold all the data from    
            the old memory, the most recent data are copied only.    
        
    '''    


    def __init__(self, size = None, type = MemoryType.ROTATE, seed = None, verbose = C_VERBOSE_NONE):
        '''
        summary: 
            Class constructor. Creates the memory (deque python object).
    
        Args:
            size: int or None
                The memory size. If None then the memory is not bounded.
                
            type: MemoryType
                The memory type, which defines the behavior when the maximum
                length is reached.
                
            seed: int or None
                Optional Seed to be used with the numpy random generator, for
                results reproducability.
                                
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
            print('\nMemory Object initialization (size = ', size, ', type = ', type, 
                  ', seed = ', seed, ')', sep = '')

        #Initialize the numpy random numbers generator
        if seed is not None:
            np.random.seed(seed)

        #Memory Size should be a valid integer or None
        if (isinstance(size, int) and size > 0) or (size is None):
            self.__size = size        
        else:
            print('ERROR: class Memory, \'', size, '\' is not a valid Memory Size, script exits.', sep = '')
            exit(-1)
   
        self.__memory = deque(maxlen = self.__size)
              
        #Memory Type should have one of the supported values
        try:
            self.__type = MemoryType(type)    
        except ValueError:
            print('ERROR: class Memory, \'', type, '\' is not a valid Memory Type, script exits.', sep = '')
            exit(-1)
            
        #Public attribute which holds the usage percentage of the memory. If the memory is not bounded, the
        #usage percenatge is always 0.0%
        self.memory_usage = 0.0
                           
        
    def add(self, data):    
        '''
        summary: 
            Adds data in the memory.
    
        Args:
            data: object
                Object to add in the memory.
        
        Raises:
            -
            
        Returns:
            -
            
        notes:
            If Memory Type is ROTATE and Memory is full, then oldest memories are discarded.
            If Memory Type is FILL and Memory is full, then the push request is ignored.
        
        '''
        
        if self.__verbose > C_VERBOSE_INFO:
            print('Add data in memory (data = ', data, ')', sep = '')
            
        
        if self.__type == MemoryType.ROTATE or len(self.__memory) < self.__size:
            self.__memory.append(data)
            
            #If memory is bounded, update the usage percentage
            if self.__size is not None:
                self.memory_usage = len(self.__memory)*100/self.__size
        
        #Discard the request, memory is full and type is Fill
        else:
            print('Add request ignored. Memory is full and type is ', self.__type, sep = '')
            
        if self.__verbose > C_VERBOSE_INFO:
            self.__view()
                
    
    def get(self, size):    
        '''
        summary: 
            Returns a batch of random memories from the memory.
        
        Args:
            size: integer
                The number of memories to be returned.
                
        Raises:
            -
                
        Returns:
            memories: numpy array
                The random memories.
            
        notes:
            -
        
        '''
        
        if self.__verbose > C_VERBOSE_INFO:
            print('Get random Memories (size = ', size, ')', sep = '')
        
        #Memory is empty
        if len(self.__memory) == 0:
            return None
        
        #If memories requested > from the memories exist in the dequeu, return all the existing memories
        return np.array([self.__memory[i] for i in np.random.random_integers(low = 0, high = len(self.__memory) - 1, 
                        size = min(size, len(self.__memory)))])
                        
            
    def __view(self):    
        '''
        summary: 
            Prints the contents of the memory.
    
        Args:
            -
            
        Raises:
            -
                
        Returns:
            -
            
        notes:
            -
        
        '''
        
        print('Memory Content: ', self.__memory, sep = '')
            
    
    def resizeMemory(self, new_size):
        '''
        summary: 
            Resizes the memory to the new_size. The data from the old memory are 
            copied to the new memory. In case the new_size cannot hold all the data from
            the old memory, the most recent data are copied only.
    
        Args:
            -
        
        Raises:
            -
            
        Returns:
            -
            
        notes:
            -
        
        '''
        
        if self.__verbose > C_VERBOSE_INFO:
            print('Resize memory (current_size = ', self.__size, ', new_size =  ', new_size, ')', sep = '')
           
        new_memory = deque(maxlen = new_size)
        
        #Copy the data from the old memory to the new one
        for i in range(new_size):
            new_memory.append(self.__memory[len(self.__memory) - new_size + i])
            
        #Reset the memory object
        self.__memory = new_memory
        self.__size = new_size
        