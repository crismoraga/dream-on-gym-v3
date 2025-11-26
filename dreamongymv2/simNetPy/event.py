# -*- coding: utf-8 -*-
"""
Created on Mon Apr 25 11:59:20 2022

@author: redno
"""

import enum


class EventType(enum.Enum):
    """EventType
        NoData = 1
        Arrive = 2
        Departure = 3
    """
    NoData = 1
    Arrive = 2
    Departure = 3


class Event:
    """Event
    Class used in simulation, there may be more than one associated with a connection event.
    """
    __eventType = EventType.NoData
    __time = -1
    __idConnection = -1

    def __init__(self, eventType, time : int, idConnection):
        """Constructor
        Default values:
            Event Type : eventType : Arrive = 2 or Departure = 3
            Time: Integer, in milisecond.
            IdConnection: Integer greater than 0
        """
        self.__eventType = eventType
        self.__time = time
        self.__idConnection = idConnection

    def getTime(self):
        """Get Time
            Returns:
                Time of Event.
        """
        return self.__time

    def getType(self):
        """Get Type
            Returns: 
                Type of Event.
        """
        return self.__eventType

    def getIdConnection(self):
        return self.__idConnection

    ''' Getter eventType '''
    @property
    def eventType(self):
        return self.__eventType

    ''' Setter eventType '''
    @eventType.setter
    def evenType(self,evenType):
        self.__eventType = evenType

    ''' Getter time '''
    @property
    def time(self):
        return  self.__time

    @time.setter
    def time(self,time):
        self.__time = time
    
    ''' Getter idConnection '''
    @property
    def idConnection(self):
        return self.__idConnection

    @idConnection.setter
    def idConnection(self,idConnection):
        self.__idConnection = idConnection