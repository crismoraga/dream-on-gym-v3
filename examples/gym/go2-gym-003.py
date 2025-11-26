# -*- coding: utf-8 -*-

from re import A
import gym
from dreamongym.simNetPy import *
from dreamongym.gym_basic import *
import os
import numpy as np


from stable_baselines3 import A2C
#from stable_baselines3.common.policies import MlpPolicy
#from stable_baselines.common.vec_env import DummyVecEnv


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
    conexiones = env.getSimulator().numberOfConnections
    #obs = np.array([-10,-5,-3])
    obs = env.observation_space.sample()
    #print("obs type state ", type(obs))
    return obs
    #return 10


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
    slot_selected = action[0]
    route = action[1]
    
    numberOfSlots = b.getNumberofSlots(0)
    link_ids = path[src][dst][route] #nodo inicio, nodo destino, ruta elegida 
    general_link = []
    for _ in range(n.getLink(0).getSlots()):
        general_link.append(False)
    for link in link_ids:
        link = n.getLink(link.id)
        for slot in range(link.getSlots()):
            general_link[slot] = general_link[slot] or link.getSlot(
                slot)
    currentNumberSlots = 0
    currentSlotIndex = slot_selected
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



env = gym.make("rlonenv-v0")

env.reset()


high = np.array([
    100,
    5,
    4
    ], dtype=np.int32,)

low = np.array([
    0,
    0,
    0
    ], dtype=np.int32,)


#definir espacio (?)
env.action_space = gym.spaces.MultiDiscrete(np.array([100,5]))
#env.observation_space = gym.spaces.Discrete(500)
#env.observation_space = gym.spaces.MultiDiscrete(np.array([100,5]))
env.observation_space = gym.spaces.Box(low,high, dtype=np.int32)

print("sample action: ", env.action_space.sample())
print(" observation shape: ", env.observation_space.shape)
print("sample observation: ", env.observation_space.sample())

#Set funcion recompensa
env.setRewardFunc(reward)
#Set estados(?)
env.setStateFunc(state)
#Definir topologia y rutas
env.initEnviroment(fileDirectory + "/BDM_Eurocore.json", fileDirectory + "/BDM_Eurocore_rutas.json")
#Cantidad de conexiones para tener el simulador corriendo 
env.getSimulator()._goalConnections = 100
env.getSimulator().setAllocator(first_fit_algorithm)
env.getSimulator().init()


env.start()
env.getSimulator().setAllocator(agent_algorithm)

model = A2C("MlpPolicy", env, verbose=False)
model.learn(total_timesteps=10)


#obs = np.array([100,5,3])

obs=env.observation_space.sample()


for i in range(10):
    # print("------------predict--------------" , i)
    #obs=env.observation_space.sample() 
    # print('obs type', type(obs))
    # print('obs type', obs)
    action, _states = model.predict(obs)
    #print('action',action)
    obs, rewards, dones, info = env.step(action)
    print(action, rewards, env.getSimulator().getBlockingProbability())