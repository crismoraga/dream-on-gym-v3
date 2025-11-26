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



env = gymnasium.make("rlonenv-v0")

env.setRewardFunc(reward)
env.setStateFunc(state)
env.initEnviroment(fileDirectory + "/NSFNet.json", fileDirectory + "/routes.json")
env.getSimulator()._goalConnections = 10
env.getSimulator().setAllocator(first_fit_algorithm)
env.getSimulator().init()

# Optional: PPO2 requires a vectorized environment to run
# the env is now wrapped automatically when passing it to the constructor
# env = DummyVecEnv([lambda: env])

env.start()
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=25000)
model.save("ppo_demo1")

del model

model = PPO.load("ppo_demo1")
obs, info = env.reset()
for i in range(1000):
    action, _states = model.predict(obs)
    obs, rewards, terminated, truncated, info = env.step(action)
    print(action, rewards, env.getSimulator().getBlockingProbability())
    #env.render()



    
