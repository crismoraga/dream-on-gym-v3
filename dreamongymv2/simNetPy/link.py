# -*- coding: utf-8 -*-
"""
Created on Mon Jan 24 14:37:03 2022

@author: redno
"""

class Link:
    """Link
    Class that connects between two nodes, has an id, size, slot number and identifies the source node and the destination node.
    """
    __id = -1
    __length = 1
    __slots = None
    __src = -1
    __dst = -1

    def __init__(self, id=-1, length=1, slots=-1, bands=None):
        """Constructor
            Initializes the slots per band, also leaving the selected band as "NoBand" by default.
            Default:
                id source node: -1
                id destiny node: -1
        """
        self.__id = id
        self.__length = length
        self.__slots = {}
        self.__slots["NoBand"] = []
        self.__slots["L"] = []
        self.__slots["C"] = []
        self.__slots["S"] = []
        self.__slots["E"] = []
        self.__slots["O"] = []
        self.__bandSelected = "NoBand"

        if bands is None:
            for _ in range(slots):
                self.__slots['NoBand'].append(False)
        else:
            for key in bands:
                for _ in range(bands[key]):
                    self.__slots[key].append(False)       
        self.__src = -1
        self.__dst = -1

    def setSlots(self, slots, band = None):
        """SetSlots
            It adjusts the number of slots available on the link, in a given band.
            Args:
                slots: Configure the quantity of nodes in the link in some band.
                band: No Band @Default None
        """
        # Search if slots are used.
        if band is None:
            slotsAux = self.__slots[self.__bandSelected]
        else:
            slotsAux = self.__slots[band]
        
        for slot in slotsAux:
            if slot == True:
                break

        if slots > len(slotsAux):
            for x in range(slots-len(slotsAux)):
                slotsAux.append(False)

        else:
            for x in range(len(slotsAux)-slots):
                del slotsAux[0]
                
    def getBands(self):
        """getBands
           Returns:
                Gets the array of slots by bands.
        """
        bands = []
        for key in range(self.__slots):
            if self.__slots[key] > 0:
                bands.append(key)
        return bands
                
    def isBandEnabled(self, band):
        """isBandEnabled
        Returns:
            If the band is available.
        """
        for key in range(self.__slots):
            if (key == band) and (self.__slots[key] > 0):
                return True
        return False

    def getSlots(self, band = None):
        """getSlots
            Returns:
                Array of slots in some band. If the band None, return the bandSelected slot array.
        """
        if band is None:
            return len(self.__slots[self.__bandSelected])
        else:
            return len(self.__slots[band])

    def getSlot(self, idSlot, band = None):
        """getSlot
            Returns:
                True or false, if the slot is used or not.
        """
        if band is None:
            return self.__slots[self.__bandSelected][idSlot]
        else:
            return self.__slots[band][idSlot]
        
    def info(self, band = None):
        """info
            Print a pretty form to see the used slots of the link 
        """
        print("ID: ", self.__id)
        print("Lenght: ", self.__length)
        print("See the state of the slots")
        if band is None:
            for slot in self.__slots[self.__bandSelected]:
                print(slot)
        else:
            for slot in self.__slots[band]:
                print(slot)
        print("Source: ", self.__src, " - Destiny: ", self.__dst)


    ''' 
        Spectrum band get and set default functions 
        Set value for band with object.band = VALUE
        Then call object.slots to get the slots array  
    '''
    @property
    def bandSelected(self):
        return self.__bandSelected
    
    @bandSelected.setter
    def bandSelected(self,band=None):
        if band is None:
            self.__bandSelected = "NoBand"
        else:
            self.__bandSelected = band

    ''' '''
    @property
    def slots(self):
        return self.__slots[self.__bandSelected]

    @slots.setter
    def slots(self,band,slots):
        self.__slots[band] = slots
            

    ''' '''
    @property
    def id(self):
        return self.__id
    
    @id.setter
    def id(self,id):
        self.__id = id

    ''' '''
    @property
    def length(self):
        return self.__length

    @length.setter
    def length(self,length):
        self.__length = length

    ''' '''
    @property
    def src(self):
        return self.__src

    @src.setter
    def src(self,src):
        self.__src = src

    ''' '''
    @property
    def dst(self):
        return self.__dst

    @dst.setter
    def dst(self,dst):
        self.__dst = dst