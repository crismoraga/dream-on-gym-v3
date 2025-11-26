# -*- coding: utf-8 -*-
"""
Created on Mon Jun 13 16:05:35 2022

@author: redno
"""

import gym
from dreamongym.simNetPy import *
from dreamongym.gym_basic import *
import os

from stable_baselines import PPO2
#from stable_baselines import A2C
#from stable_baselines import ACKTR
#from stable_baselines import DQN
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv

##Llamados a funciones
def reward():
    value = env.getSimulator().lastConnectionIsAllocated()
    if (value.name == Controller.Status.Not_Allocated.name):
        value = -1
    else:
        value = 1
    return value

def state():
    return 3

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



env = gym.make("rlonenv-v0")

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
model = PPO2(MlpPolicy, env, verbose=False)
model.learn(total_timesteps=10000)

obs = env.reset()
for i in range(1000):
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    print(action, rewards, env.getSimulator().getBlockingProbability())
    #env.render()



    