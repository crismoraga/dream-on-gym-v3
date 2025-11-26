# -*- coding: utf-8 -*-
"""
Created on Tue Feb 22 16:08:34 2022

@author: redno
"""
import json

class BitRate:
    """BitRate
    """
    __bitRate = 0.0
    __modulation = None
    __slots = None
    __reach = None
    
    def __init__(self, bitRate=None):
        """Constructor
            The BitRate is a component to decide the distanse of the technologie and the slots enabled.
        """
        self.__bitRate = bitRate
        self.__modulation = []
        self.__slots = []
        self.__reach = []
    
    def addModulation(self, modulation, slots : int, reach):
        """addModulation
            Method to add a modulation with some slots and define the distance.
            Args:
                modulation: Is a number that describe the modulation type.
                slots: Quantity of slots
                reach: Distance capable
        """
        self.__modulation.append(modulation)
        self.__slots.append(slots)
        self.__reach.append(reach)
    
    def getModulation(self, position):
        if (position >= len(self.__modulation)):
            raise ("Bitrate "+self.__bitRate+" does not have more than "+len(self.__modulation)+" modulations.")
        return self.__modulation[position]
    
    def getNumberofSlots(self, position):
        if (position >= len(self.__slots)):
            raise("Bitrate "+self.__bitRate+" does not have more than "+len(self.__slots)+" slots.")
        return self.__slots[position]
    
    def getReach(self, position):
        if (position >= len(self.__reach)):
            raise ("Bitrate "+self.__bitRate+" does not have more than "+len(self.__reach)+" reach.")
        return self.__reach[position]
    
    def readBitRateFile(self, fileName):
        """readBitRateFile
            Charge the configuration of bitRate network from file.
            Args:
                filename: Is the path of is stored the file, include the filename.
        """
        with open(fileName) as json_file:
            info = json.load(json_file)
            #json_strings = json.dumps(data, indent=4)
            bitsRate = []
            for tag in info:
                bitRate = BitRate(tag)
                for name in info[tag]:
                    for modulation in name:
                        bitRate.addModulation(modulation,name[modulation]['slots'],name[modulation]['reach'])
                bitsRate.append(bitRate)
            return bitsRate

    ''' '''
    @property
    def bitRate(self):
        return self.__bitRate
    
    @bitRate.setter
    def bitRate(self,bitRate):
        self.__bitRate = bitRate

    ''' '''
    @property
    def modulation(self):
        return self.__modulation
    
    @modulation.setter
    def modulation(self,modulation):
        self.__modulation.append(modulation)

    ''' '''
    @property
    def slots(self):
        return self.__slots
    
    @slots.setter
    def slots(self,slots):
        self.__slots.append(slots)

    ''' '''
    @property
    def reach(self):
        return self.__reach

    @reach.setter
    def reach(self,reach):
        self.__reach.append(reach)