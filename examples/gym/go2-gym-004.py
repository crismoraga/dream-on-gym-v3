# -*- coding: utf-8 -*-
"""
Created on Mon Jun 13 16:05:35 2022

@author: redno
"""

from dreamongymv2.simNetPy import *
from dreamongymv2.gym_basic import *
import os
import sys
import gymnasium
sys.modules["gym"] = gymnasium

from stable_baselines3 import DQN
from stable_baselines3 import PPO
##Llamados a funciones
def reward():
    value = env.getSimulator().lastConnectionIsAllocated()
    if (value.name == Controller.Status.Not_Allocated.name):
        #print(value.name)
        value = -1
    else:
        value = 1
    return value

def state():
    return 2

def first_fit_algorithm(src: int, dst: int, b: BitRate, c: Connection, n: Network, path, action):
    numberOfSlots = b.getNumberofSlots(0)
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

def agent_algorithm(src: int, dst: int, b: BitRate, c: Connection, n: Network, path, action):
    numberOfSlots = b.getNumberofSlots(0)
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
    currentSlotIndex = action
    if numberOfSlots+currentSlotIndex < len(general_link):
        for k in range(numberOfSlots):
            if not general_link[currentSlotIndex + k]:
                currentNumberSlots += 1
            else:
                currentNumberSlots = 0
                break
            if currentNumberSlots == numberOfSlots:
                for k in link_ids:
                    c.addLink(
                        k, fromSlot=currentSlotIndex, toSlot=currentSlotIndex+currentNumberSlots)
                return Controller.Status.Allocated, c
    return Controller.Status.Not_Allocated, c


def agent_algorithmArrayAction(src: int, dst: int, b: BitRate, c: Connection, n: Network, path, action):
    numberOfSlots = b.getNumberofSlots(0)
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
    for j in range(len(action)):
        currentSlotIndex = action[j]
        if numberOfSlots+currentSlotIndex < len(general_link):
            for k in range(numberOfSlots):
                if not general_link[currentSlotIndex + k]:
                    currentNumberSlots += 1
                else:
                    currentNumberSlots = 0
                    break
                if currentNumberSlots == numberOfSlots:
                    for k in link_ids:
                        c.addLink(
                            k, fromSlot=currentSlotIndex, toSlot=currentSlotIndex+currentNumberSlots)
                    return Controller.Status.Allocated, c
    return Controller.Status.Not_Allocated, c

absolutepath = os.path.abspath(__file__)
fileDirectory = os.path.dirname(absolutepath)



env = gymnasium.make("rlonenv-v0")
env.action_space = gymnasium.spaces.Discrete(40)
env.setRewardFunc(reward)
env.setStateFunc(state)
env.initEnviroment(fileDirectory + "/NSFNet.json", fileDirectory + "/routes.json")
env.getSimulator()._goalConnections = 10000
env.getSimulator().setAllocator(first_fit_algorithm)
env.getSimulator().init()
env.start()
env.getSimulator().setAllocator(agent_algorithm)
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=10000)
model.save("ppo_demo2")

del model

model = PPO.load("ppo_demo2")

obs, info = env.reset()

#
# START PREDICT
#
env.initEnviroment(fileDirectory + "/BDM_EON.json", fileDirectory + "/BDM_EON_rutas.json")
env.getSimulator()._goalConnections = 10000
env.getSimulator().init()
env.getSimulator().setAllocator(first_fit_algorithm)
env.start()
env.getSimulator().setAllocator(agent_algorithmArrayAction)
for i in range(100):
    array_actions = []
    for _ in range(10):
        action, _states = model.predict(obs)
        array_actions.append(action)
    obs, rewards, terminated, truncated, info = env.step(array_actions)
    print(array_actions, rewards, env.getSimulator().getBlockingProbability())    
