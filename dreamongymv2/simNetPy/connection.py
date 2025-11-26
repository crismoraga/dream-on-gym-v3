# -*- coding: utf-8 -*-
"""


@author: redno
"""

class Connection:
    """Connection
        Indicates which links and how many slots a connection between two users has.
    """
    __id : int
    __links : None
    __slots : None
    __bandSelected : str
    
    def __init__(self, id : int):
        """The constructor
            Args:
                id: ID of Connection, Integer 
            Default values:
            Band Selected: "NoBand"
        """
        self.__id = id
        self.__links = []
        self.__slots = []
        self.__bandSelected : str = "NoBand"
    
    def addLink(self, idLink : int, slots: int):
        """addLink
            Adds a link to the connection with a slot amount.
        Args:
            int idLink
            int slots
        
        """
        self.__links.append(idLink)
        self.__slots.append(slots)
        
    def addLink(self, idLink, fromSlot: int, toSlot : int):
        """addLink
            Adds a link to the connection with a range of slots amount.
            Args:
                int idLink
                int fromSlot
                int toSlot
        """
        self.__links.append(idLink)
        lSlots = []
        for i in range(fromSlot, toSlot):
            lSlots.append(i)
        self.__slots.append(lSlots)
        
    @property
    def bandSelected(self):
        """bandSelected
            This property changes the selected band, it can be:
            1) NoBand
            2) C
            3) L
            4) S
            5) E
        """
        return self.__bandSelected
    
    @bandSelected.setter
    def bandSelected(self, bandSelected : str):
        self.__bandSelected = bandSelected

    ''' Id getter & setter '''
    @property
    def id(self):
        return self.__id

    @id.setter
    def id(self,id):
        self.__id = id

    ''' Links getter & setter '''
    @property
    def links(self):
        return self.__links

    @links.setter
    def links(self,idLinks):
        #self.__links = []
        self.__links = idLinks       # Asigna todos los links en la conexión en vez de ir uno por uno 

    ''' Slots getter & setter '''
    @property
    def slots(self):
        return self.__slots

    @slots.setter
    def slots(self,slots):
        #self.__slots = []
        self.__slots = slots       # Asigna todos los slots de cada enlace en la conexión  