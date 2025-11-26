# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import torch as th
from sb3_contrib import TRPO
from sb3_contrib.trpo import MlpPolicy as MLP_TRPO
from dreamongymv2.simNetPy import *
from dreamongymv2.gym_basic import *
import os
import sys
import numpy as np
import gymnasium
sys.modules["gym"] = gymnasium

bands = ['L','C','S','E']

episodeRewardAccum = 0
iteration = 0
episodeIteration = 100
episodeRewardsArray = []
##Llamados a funciones
def reward():
    global episodeRewardAccum
    global episodeIteration
    global episodeRewardsArray
    global iteration
    iteration = iteration + 1
    if (iteration % episodeIteration) == 0:
        episodeRewardsArray.append(episodeRewardAccum)
        episodeRewardAccum = 0
    value = env.getSimulator().lastConnectionIsAllocated()
    if (value.name == Controller.Status.Not_Allocated.name):
        #print(value.name)
        value = -1
    else:
        value = 1
    episodeRewardAccum = episodeRewardAccum + value
    return value

def state():
    connectionEvent = env.getSimulator().connectionEvent
    route = env.getSimulator().controller.path[connectionEvent.source][connectionEvent.destination][0]
    slotsQuantity = 0
    for i in range(4):
        slotsQuantity = slotsQuantity + route[0].getSlots(getBand(i))
    obs = [0] * slotsQuantity
    slotInit = 0
    for i in range(len(route)):
        slotIterator = 0
        for j in range(4):
            slotInit = slotIterator
            for k in range(slotInit, slotInit + route[i].getSlots(getBand(j))):
                slotIterator = slotIterator + 1
                if route[i].getSlot(k-slotInit,getBand(j)):
                    obs[k] = obs[k] + 1
    return obs

def stateAllNetwork():
    connectionEvent = env.getSimulator().connectionEvent
    routes = env.getSimulator().controller.path[connectionEvent.source][connectionEvent.destination]
    slotsQuantity = 0
    for i in range(4):
        slotsQuantity = slotsQuantity + routes[0][0].getSlots(getBand(i))
    obs = [0] * slotsQuantity
    slotInit = 0
    for i in range(len(env.getSimulator().controller.network.links)):
        slotIterator = 0
        for z in range(4):
            slotInit = slotIterator
            for k in range(slotInit, slotInit + env.getSimulator().controller.network.links[i].getSlots(getBand(z))):
                slotIterator = slotIterator + 1
                if env.getSimulator().controller.network.links[i].getSlot(k-slotInit,getBand(z)):
                    obs[k] = obs[k] + 1
    return obs

def first_fit_algorithm(src: int, dst: int, b: BitRate, c: Connection, n: Network, path, action):
    numberOfSlots = b.getNumberofSlots(0)
    link_ids = path[src][dst][0]
    c.bandSelected = "C"
    general_link = []
    for _ in range(n.getLink(0).getSlots(c.bandSelected)):
        general_link.append(False)
    
    for link in link_ids:
        link = n.getLink(link.id)
        link.bandSelected = c.bandSelected
        for slot in range(link.getSlots(c.bandSelected)):
            general_link[slot] = general_link[slot] or link.getSlot(slot,c.bandSelected)
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

def getBand(actionBand):
    if (actionBand == 0):
        return 'C'
    elif (actionBand == 1):
        return 'L'
    elif (actionBand == 2):
        return 'S'
    elif (actionBand == 3):
        return 'E'
    
    return 'NoBand'
def agent_algorithm(src: int, dst: int, b: BitRate, c: Connection, n: Network, path, action):
    band = getBand(action[0])
    c.bandSelected = band
    numberOfSlots = b.getNumberofSlots(0)
    actionSpace = len(path[src][dst])
    link_ids = path[src][dst][0]
    general_link = []
    for _ in range(n.getLink(0).getSlots(band)):
        general_link.append(False)
    for link in link_ids:
        link = n.getLink(link.id)
        link.bandSelected = band
        for slot in range(link.getSlots()):
            general_link[slot] = general_link[slot] or link.getSlot(slot,band)
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

env = gymnasium.make("rlonenv-v0")

obs, info = env.reset() 

#definir espacio (?)
env.action_space = gymnasium.spaces.MultiDiscrete(np.array([4]))

env.observation_space = gymnasium.spaces.MultiDiscrete([100] * 2720)

#Set funcion recompensa
env.setRewardFunc(reward)
#Set estados(?)
env.setStateFunc(state)
#env.setStateFunc(stateAllNetwork)
#Definir topologia y rutas
env.initEnviroment(fileDirectory + "/NSFNet_4_bands.json", fileDirectory + "/routes.json")
#Cantidad de conexiones para tener el simulador corriendo 
env.getSimulator().goalConnections = 100000
env.getSimulator().setMu(1)
env.getSimulator().setLambda(1000)
env.getSimulator().setAllocator(first_fit_algorithm)
env.getSimulator().init()


env.start()
env.getSimulator().setAllocator(agent_algorithm)

policy_args = dict(net_arch=5*[128], activation_fn=th.nn.ReLU)
model = TRPO(MLP_TRPO, env, verbose=0, tensorboard_log="./tb/TRPO-DeepRBMLSA-v0/", policy_kwargs=policy_args, gamma=.95)
model.learn(total_timesteps=1000, log_interval=4)

model.save('TRPO120720231300')
#model.load('TRPO100720231300')
obs, info = env.reset() 
rewardsArray = []
rewardTemp = 0
iterationAccum = 100
_state = [0] * 2720
for i in range(1000):
    if (i % iterationAccum) == 0:
        rewardsArray.append(rewardTemp)
        rewardTemp = 0
    action, _states = model.predict(_state)
    _state, _reward, terminated, truncated, info = env.step(action)
    rewardTemp = rewardTemp + _reward
print(episodeRewardsArray)
print(rewardsArray)
#plt.plot(range(len(rewardsArray)), rewardsArray)
