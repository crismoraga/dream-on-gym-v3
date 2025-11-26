# -*- coding: utf-8 -*-
"""


@author: redno
"""
# from .allocator import Allocator
import string

from .network import Network
from .connection import Connection
import enum
import json

### modify access to private variables of Connections & Network classes


class Controller:
    """Controller

    Object that is tasked with controlling the allocation and deallocation of connection slots.
    """

    def __init__(self, network : Network = None):
        """The constructor
            Args:
                network: Network Object
            Default values:
            Source : None
            Destination: None
            BitRate: None
            IdConnection: None
        """
        self.__connections = []
        self.__allocator = None
        self.__network = network
        self.__path = []

    ''' Check privacy of status // __status return error with N_A when calling getter in Simulator '''
    Status = enum.Enum('Status','Allocated Not_Allocated N_A')

    def assignConnection(self, src, dst, bitRate, idConnection : int, action = None):
        """assignConnection
        Args:
            src: Source User
            dst: Destination User
            bitRate: Bit Rate 
            idConnection: ID Connection
            action: The agent's decision value can be of various types. For example: an integer, a float, an array of values, etc.

        According to what the allocator algorithm delivered, the slots requested in the connection are assigned on all links.
        """
        con = Connection(idConnection)

        rtnAllocation, con = self.__allocator(
            src, dst, bitRate, con, self.__network, path=self.__path, action=action)
        if (rtnAllocation.name == Controller.Status.Allocated.name):
            self.__connections.append(con)
            for j in range(0, len(con.links)):
                self.__network.useSlot(con.links[j].id, con.slots[j][0], con.slots[j][0]+len(con.slots[j]), con.bandSelected)
        return rtnAllocation

    def unassignConnection(self, idConnection):
        """unassignConnection
        Args:
            idConnection: ID Connection

            According to what the allocator algorithm delivered, the slots requested in the connection are unassigned on all links.
        """
        for i in range(0, len(self.__connections)):
            if (self.__connections[i].id == idConnection):
                con = self.__connections[i]
                for j in range(0, len(con.links)):
                    self.__network.unuseSlot(con.links[j].id, con.slots[j][0], con.slots[j][0]+len(con.slots[j]), con.bandSelected)
                
                self.__connections.pop(i)
                break
        return 0

    def setPaths(self, filename):
        """setPaths
        Args:
            filename : Path URL to read the route of the network

        This method reads the routes file of a network.
        """
        with open(filename) as json_file:
            filePaths = json.load(json_file)
        numberOfNodes = self.__network.getNumberOfNodes()
        self.__path = []

        for i in range(0, numberOfNodes):
            self.__path.append([])
            for j in range(0, numberOfNodes):
                self.__path[i].append([])
        routesNumber = len(filePaths['routes'])

        for i in range(0, routesNumber):
            # Numero de caminos
            #     "paths": [
            #     [0, 1],
            #     [0, 2, 1],
            #     [0, 7, 6, 4, 3, 1]
            #   ]
            pathsNumber = len(filePaths['routes'][i]['paths'])

            # "src": 0,
            src = filePaths['routes'][i]['src']

            # "dst": 1,
            dst = filePaths['routes'][i]['dst']
            for j in range(0, pathsNumber):
                self.__path[src][dst].append([])
            for j in range(0, pathsNumber):
                # ubicación de un camino
                #   "paths": [
                #     [0, 1],      <---- este por ejemplo
                #     [0, 2, 1],
                #     [0, 7, 6, 4, 3, 1]
                #   ]
                nodesPathNumber = len(filePaths['routes'][i]['paths'][j])
                lastNode = nodesPathNumber - 1
                for k in range(0, lastNode):
                    actNode = filePaths['routes'][i]['paths'][j][k]
                    nextNode = filePaths['routes'][i]['paths'][j][k+1]
                    idLink = self.__network.isConnected(actNode, nextNode)

                    self.__path[src][dst][j].append(
                        self.__network.getLink(idLink))
        return 0

    def getConnection(self, idConnection):
        """getConnection
        Args:
            idConnection: Integer ID of Connection

        Return the Object Connection
        """
        for i in range(len(self.__connections)):
            if self.__connections[i].id == idConnection:
                return self.__connections[i]
        return None
    ''' Connection List getter & setter '''
    @property
    def connections(self):
        return self.__connections

    @connections.setter
    def connections(self,connection):
        self.__connections.append(connection)

    ''' Allocation Function getter & setter '''
    @property
    def allocator(self):
        return self.__allocator

    @allocator.setter
    def allocator(self, allocator):
        self.__allocator = allocator

    ''' Network getter & setter '''
    @property
    def network(self):
        return self.__network

    @network.setter
    def network(self, network):
        self.__network = network 

    ''' Array Path of Routes getter & setter '''
    @property
    def path(self):
        return self.__path

    ''' '''
    #@property
    #def Status(self):
    #    return self.__Status