# -*- coding: utf-8 -*-
"""
Created on Mon Jan 24 17:51:19 2022

@author: redno
"""
from .filemanager.readerJson import Reader
import sys
sys.path.append('../')


class Network:
    """Network
        This class contains all the nodes and links in the network and their connections, from here you can access every corner of the network.
    """

    def __init__(self, filename : str):
        """Constructor
            This creates the network based on a file and initializes the arrays
            Args:
                filename : Is a path directory
        """
        self.__linkCounter = 0
        self.__nodeCounter = 0
        self.__nodes = []
        self.__links = []
        self.__linksIn = []
        self.__linksOut = []
        self.__nodesIn = []
        self.__nodesOut = []
        self.__nodesIn.append(0)
        self.__nodesOut.append(0)

        # Open JSON File
        j = Reader()
        j.readNetwork(filename, self.nodes, self.links)
        # Number of Nodes
        self.__nodeCounter = len(self.nodes)
        self.__linkCounter = len(self.links)
        outCount = 0
        inCount = 0
        for i in range(self.nodeCounter):
            for j in range(self.linkCounter):
                if (i == self.links[j].src):
                    self.__linksOut.append(self.links[j])
                    outCount = outCount + 1
                if (i == self.links[j].dst):
                    self.__linksIn.append(self.links[j])
                    inCount = inCount + 1
            self.nodesOut.append(outCount)
            self.nodesIn.append(inCount)

    def addNode(self, node):
        """addNode
            Method to add nodes in the network.
            Args:
                node : Object node
        """
        if (node._id != self.__nodeCounter):
            raise("Cannot add a Node to this network with Id mismatching node counter.")
        self.__nodeCounter = self.nodeCounter + 1
        self.nodes.append(node)
        self.nodesIn.append(0)
        self.nodesOut.append(0)

    def addLink(self, link):
        """addLink
            Method to add links in the network.
            Args:
                link: Object Link  
        """
        if (link._id != self.linkCounter):
            raise("Cannot add a Link to this network with Id mismatching link counter.")
        self.__linkCounter = self.linkCounter + 1
        self.links.append(link)

    def connect(self, src : int, linkPos : int, dst : int):
        """connect
            Connect two nodes with a link
            Args:
                src : Source ID
                linkPos: Link ID
                dst : Destiny ID            
        """
        if (src < 0 or src >= self.__nodeCounter):
            raise("Cannot connect src "+src +
                  " because its ID is not in the network. Number of nodes in network: "+self.__nodeCounter)
        if (dst < 0 or dst >= self.__nodeCounter):
            raise("Cannot connect dst "+dst +
                  " because its ID is not in the network. Number of nodes in network: "+self.__nodeCounter)
        if (linkPos < 0 or linkPos >= self.__linkCounter):
            raise("Cannot use link "+linkPos +
                  " because its ID is not in the network. Number of links in network: "+self.__linkCounter)
        self.__linksOut.insert(
            self.linksOut[0] + self.nodesOut[src], self.links[linkPos])
        for n in range(self.nodesOut[0] + src + 1, self.nodesOut[-1]):
            self.nodesOut[n] = self.nodesOut[n] + 1
        self.__linksIn.insert(
            self.linksIn[0] + self.__nodesIn[dst], self.links[linkPos])
        for n in range(self.nodesIn[0] + dst + 1, self.nodesIn[-1]):
            self.nodesIn[n] = self.nodesIn[n] + 1
        self.links[linkPos]._src = src
        self.links[linkPos]._dst = dst

    def isConnected(self, src : int, dst : int):
        """isConnected
            Args:
                src : Source ID
                dst : Destiny ID
            Returns:
                if the source has a connection to destiny.
        """
        for i in range(self.__nodesOut[src], self.__nodesOut[src+1]):
            for j in range(self.__nodesIn[dst], self.__nodesIn[dst+1]):
                if(self.__linksOut[i].id == self.__linksIn[j].id):
                    return self.__linksOut[i].id
        return -1

    def useSlot(self, linkPos, slotFrom, slotTo = None, bandSelected = "NoBand"):
        """useSlot
            Take a free slots from slotFrom to slotTo into a BandSelected.
            Change all the values to True
            Args:
                linkPos: Link ID
                slotFrom: Initial Range Slot required
                slotTo: End Range Slot required
                bandSelected: Band Selected L, C, S, E, O or NoBand is the default value.
        """
        self.links[linkPos].bandSelected = bandSelected
        if (slotTo is None):
            if (linkPos < 0 or linkPos > self.__linkCounter):
                raise("Link position out of bounds.")
            if (self.links[linkPos].slots[slotFrom] == True):
                raise("Bad assignation on slot",slotFrom)
            self.links[linkPos].slots[slotFrom] = True
        else:
            self.validateSlotFromTo(linkPos, slotFrom, slotTo)
            for i in range(slotFrom, slotTo):
                self.links[linkPos].slots[i] = True

    def unuseSlot(self, linkPos, slotFrom, slotTo = None, bandSelected = "NoBand"):
        """unuseSlot
            Take a free slots from slotFrom to slotTo into a BandSelected.
            Change all the values to False
            Args:
                linkPos: Link ID
                slotFrom: Initial Range Slot required
                slotTo: End Range Slot required
                bandSelected: Band Selected L, C, S, E, O or NoBand is the default value.    
        """
        self.links[linkPos].bandSelected = bandSelected
        if (slotTo is None):
            if (linkPos < 0 or linkPos > self.__linkCounter):
                raise("Link position out of bounds.")
            self.links[linkPos].slots[slotFrom] = False
        else:
            self.validateSlotFromTo(linkPos, slotFrom, slotTo)
            for i in range(slotFrom, slotTo):
                self.links[linkPos].slots[i] = False

    def isSlotUsed(self, linkPos, slotPos):
        """isSlotUsed
            Indicates whether a certain slot is used or not, on a given link.
            Args:
                linkPos: link ID
                slotPos: Slot ID
            Returns:
                True if slot is used, false in other case.
        """
        if (linkPos < 0 or linkPos >= self.linksCounter):
            raise("Link position out of bounds.")
        if (slotPos < 0 or slotPos >= self.links[linkPos].getSlots()):
            raise("slot position out of bounds.")
        return self.links[linkPos].slots[slotPos]

    def validateSlotFromTo(self, linkPos, slotFrom, slotTo):
        """validateSlotFromTo
            Validates whether the slots exist within a given link.
            Args:
                linkPos: Link ID
                slotFrom: Initial of range of slots
                slotTo: End of range of slots.     
        """
        if (linkPos < 0 or linkPos >= self.linkCounter):
            raise("Link position out of bounds.")
        if (slotFrom < 0 or slotFrom >= self.links[linkPos].getSlots()):
            print("slot position out of bounds. (From Slot",slotFrom,") of ",self.links[linkPos].getSlots())
            raise("slot position out of bounds.")
        if (slotTo < 0 or slotTo > self.links[linkPos].getSlots()):
            print("slot position out of bounds. (To Slot", slotTo, ") of ",self.links[linkPos].getSlots())
            raise("slot position out of bounds.")
        if (slotFrom > slotTo):
            raise("Initial slot position must be lower than the final slot position.")
        if (slotFrom == slotTo):
            raise("Slot from and slot To cannot be equals.")

    def getNumberOfNodes(self):
        """getNumberOfNodes
            Get the count of nodes in the network
            Returns:
                Obtain the quantity of nodes in the network
        """
        return self.__nodeCounter

    def getNumberOfLinks(self):
        """getNumberOfLinks
            Get the count of links in the network
            Returns:
                Obtain the quantity of links in the network
        """
        return self.__linkCounter

    def getLink(self, idLink):
        return self.__links[idLink]

    ''' '''
    @property
    def linkCounter(self):
        return self.__linkCounter

    ''' '''
    @property
    def nodeCounter(self):
        return self.__nodeCounter

    ''' '''
    @property
    def nodes(self):
        return self.__nodes
    
    ''' '''
    @property
    def links(self):
        return self.__links

    ''' '''
    @property
    def linksIn(self):
        return self.__linksIn

    ''' '''
    @property
    def linksOut(self):
        return self.__linksOut

    ''' '''
    @property
    def nodesIn(self):
        return self.__nodesIn

    ''' '''
    @property
    def nodesOut(self):
        return self.__nodesOut