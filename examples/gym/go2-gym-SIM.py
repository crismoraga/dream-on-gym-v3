# -*- coding: utf-8 -*-
"""
Created on Mon Jun 13 16:05:35 2022

@author: redno
"""
import os
import sys
import gymnasium
sys.modules["gym"] = gymnasium
from dreamongymv2.simNetPy.simulator_finite import Simulator
from dreamongymv2.simNetPy.bitRate import BitRate
from dreamongymv2.simNetPy.connection import Connection
from dreamongymv2.simNetPy.network import Network
from dreamongymv2.simNetPy.controller import Controller

##Llamados a funciones
def first_fit_algorithm(src: int, dst: int, b: BitRate, c: Connection, n: Network, path, action):
    numberOfSlots = b.getNumberofSlots(0)
    actionSpace = len(path[src][dst])
    if action is not None:
        if action == actionSpace:
            action = action - 1
        link_ids = path[src][dst][action]
    else:
        link_ids = path[src][dst][0]
    general_link = []
    for _ in range(n.getLink(0).getSlots()):
        general_link.append(False)
    for link in link_ids:
        link = n.getLink(link.id)
        for slot in range(link.getSlots()):
            general_link[slot] = general_link[slot] or link.getSlot(
                slot)
    currentNumberSlots = 0
    currentSlotIndex = 0
    
    for j in range(len(general_link)):
        if not general_link[j]:
            currentNumberSlots += 1
        else:
            currentNumberSlots = 0
            currentSlotIndex = j + 1
        if currentNumberSlots == numberOfSlots:
            for k in link_ids:
                c.addLink(
                    k, fromSlot=currentSlotIndex, toSlot=currentSlotIndex+currentNumberSlots)
            return Controller.Status.Allocated, c
    return Controller.Status.Not_Allocated, c


absolutepath = os.path.abspath(__file__)
fileDirectory = os.path.dirname(absolutepath)

simulator = Simulator(fileDirectory + "/GermanNet.json", fileDirectory + "/GermanNet_routes.json","")
        
simulator.setGoalConnections(1000000)
#simulator.__timeOn = 1000
simulator.setRho(0.5)
simulator.setAllocator(first_fit_algorithm)
simulator.init()
simulator.run(True)




    
