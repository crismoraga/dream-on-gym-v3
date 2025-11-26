# -*- coding: utf-8 -*-
"""
Created on Tue May 17 12:41:23 2022

@author: redno
"""
from .randomVariable import RandomVariable
import math


class ExpVariable:
    """Exp Variable
    Class that returns a random number based on a random function, a seed, and a value
    """

    def __init__(self, seed=1234567, parameter1=10):
        """Constructor
            Args:
                seed: Large prime number. Default: 1234567
                parameter: Adjust de value of random number. Default: 10
        """
        if (parameter1 <= 0):
            raise("Lambda parameter must be positive.")
        self.__rn = RandomVariable(seed, parameter1)

    # dist corresponde a un objeto en c++
    def getNextValue(self):
        """getNextValue
            Returns:
                -1*log(1 - random(seed)) / param)
        """
        return -1*(math.log(1 - self.__rn.getDist(self.__rn.generator)) / self.__rn.parameter1)

    ''' Getter random number '''
    @property
    def rn(self):
        return self.__rn

    ''' Setter random number '''
    @rn.setter
    def rn(self,seed,parameter1):
        self.__rn = RandomVariable(seed, parameter1)
