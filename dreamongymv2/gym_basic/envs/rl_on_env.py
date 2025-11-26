# -*- coding: utf-8 -*-
"""@package gym_basic.envs



@author: Hermann Pempelfort
"""

import gymnasium
from dreamongymv2.simNetPy.simulator_finite import Simulator


class RlOnEnv(gymnasium.Env):
    """Enviroment for Gymnasium
    
    Tool that connects the simulator and the agent in an optical network environment
    """
    def __init__(self, action_space = 3, observation_space = 3, start_training = 1000):
        """The constructor
            Args:
            Action Space: 3
            Observation Space: 3
            Start Training: 1000 iterations
        """
        self.action_space = gymnasium.spaces.Discrete(action_space)
        self.observation_space = gymnasium.spaces.Discrete(observation_space)
        self.__simulator = None
        self.__rewardFunc = None
        self.__stateFunc = None
        self.__startTraining = start_training
        
    def start(self, verbose=False):
        """Start Function
        Initializes the simulator to the start_training stage, then proceeds to the step-by-step or reset stage.
        This action allows you to prepare the environment for agent interaction if the start_training stage is high.
        """
        self.__simulator.init()
        if self.__simulator is not None:
            self.__simulator.run(verbose)
            #for i in range(0)
    def step(self, action):
        """Step Function
        Receives the action taken by the agent.
        The function takes the value sent according to the programmed function.
        The simulator assigns or deems it appropriate based on the decision, and a benefit is obtained based on this.
        """
        if self.__simulator is not None: 
            #Se debe setear la acción tomada por el agente en el simulador.
            self.__simulator.step(action)
            self.__simulator.forwardDepartures()
            self.__simulator.createEventConnection()
            #Se debe recuperar el estado por omisión se deja 1
            if self.__stateFunc is not None:
                state = self.__stateFunc()
            else:
                state = 1
            if self.__rewardFunc is not None:
                reward = self.__rewardFunc()
            if self.__simulator.getTruncatedFunc() is not None:
                truncated = self.__simulator.TruncatedFunc()
            else:
                truncated = True
            if self.__simulator.getTerminatedFunc() is not None:
                terminated = self.__simulator.TerminatedFunc()
            else:
                terminated = True
        else:
            state = 1
            if action == 2:
                reward = 1
            else:
                reward = -1    
            truncated = True
            terminated = True
        
        info = {}
        return state, reward, terminated, truncated, info
    def reset(self, seed=None, options=None):
        """Reset Function
        Function that returns the simulation to the beginning.
        """
        #Se debe recuperar el estado al resetiar, por omisión se deja en 0
        state = 0
        info = {}
        return state, info
    
    def setStateFunc(self, func):
        """Function State
        Set the funtion state.
        """
        self.__stateFunc = func
        
    def setRewardFunc(self, func):
        """Function Reward
        Set the funtion reward.
        """
        self.__rewardFunc = func
    
    def initEnviroment(self, networkFilename="", pathFilename="", bitrateFilename=""):
        """Initialization Enviroment
        Create the Simulator Object.
        """
        self.__simulator = Simulator(networkFilename, pathFilename, bitrateFilename)
        self.__simulator.setGoalConnections(self.__startTraining)
    
    def getSimulator(self):
        """Get Simulator Object
        """
        return self.__simulator
        

        
