# DREAM-ON GYM Deep REinforcement learning freAMwork for Optical Networks

**DREAM-ON GYM v2** is a Python Framework which can be configurated as a Network Enviroment for training Agents.

This framework use the Simulator "Flex Net Sim" [Git-Lab] https://gitlab.com/DaniloBorquez/flex-net-sim


[[_TOC_]]

## Features

- Load Agents with OpenAI
- Create your own Optical Network Enviroment
- Configurate your own reward, done, state and info functions
- Configurate many parameters:
	- Start training
	- Determinate the Step function
	- Determinate the Reset function
- Elastic optical networks support
- Support for different bitrates through a JSON file
- Support for different networks through a JSON file
- Support for multiple routes through a JSON file
- Customize connection arrive/departure ratios
- Support to create your own statistics
- Customize the number of connection arrives that you want to simulate

## Download

Download the latest release from the [release list](https://gitlab.com/IRO-Team/dream-on-gym-v2.0). 

If you want to try the current in-development version (which could be unstable), you can clone this repository through the git clone command.

```
git clone git@gitlab.com:IRO-Team/dream-on-gym-v2.git
```

## Pre-Installation

### Nvidia requirements:
+ Max Versión of Nvidia CUDA Toolkit September 2018

### Windows requirements:
+ Install Microsoft Visual Build Tool 2019 or above
+ You must add two environment variables in Windows
+ MPILib C:\Program Files (x86)\Microsoft SDKs\MPI\Lib
+ MPI C:\Program Files (x86)\Microsoft SDKs\MPI
+ Install [MDI Versión 10.0] (https://www.microsoft.com/en-us/download/details.aspx?id=57467)

### Version requirements:
+ Python = [3.10.6] (https://www.python.org/downloads/release/python-3711/)
+ The last Stable-Baselines = [SB3] (https://github.com/DLR-RM/stable-baselines3)
	+ Please install from git version, download, unzip, and then into the unzipped folder: pip install .
+ The Last Tensor-flow = 2.12.0 or great
	+ For GPU users: pip install tensorflow[and-cuda]
	+ For CPU users: pip install tensorflow
+ Protobuf = 4.22.3 or great 
	+ pip install protobuf
+ Gymnasium = 0.28.1
	+ pip install gymnasium
+ pip install mpi4py (For this step, you need the Windows requirements complete installed)


## Installation

The latest release can be downloaded to install our tool, or the git code can be cloned.
Later, go to the root folder and execute the "pip install" command in the console.


## Using  Framework
 
- The Framework loads an Optical Network Environment to train Agents with Reinforcement Learning.
- This Framework can be used as an Application and to adapt the code for training Agents in Optical Networks flexibly.
- The example folder contains examples files.
- In the examples, the agent, parameters, hyperparameters, and network, among other parameters, can be modified.
- Any function can be modified, such as the reward, action, state, done, and info functions.

# Simple Example

To use the framework package using pip install execute the following command [pip](https://pip.pypa.io/en/stable/) to install dream-on-gym-v2.

```bash
pip install dream-on-gym-v2
```

## Reward Function
As an example, we use the standard reward function, which returns "1" if the connection is allocated, else "-1".
```python
def reward():
    value = env.getSimulator().lastConnectionIsAllocated()
    if (value.name == Controller.Status.Not_Allocated.name):
        value = -1
    else:
        value = 1
    return value
```


## Load Enviroment

```python
import os
import gym
from simNetGymPy import *
import que-dificl

# Get local path's
absolutepath = os.path.abspath(__file__)
fileDirectory = os.path.dirname(absolutepath)

#Create the que-dificil Enviroment
env = gym.make("rlonenv-v0")

#Set the reward function
env.setRewardFunc(reward)

#Load the simulator statements and the allocator function 'first_fit_algorithm'
env.initEnviroment(fileDirectory + "/NSFNet.json", fileDirectory + "/NSFNet_routes.json")
env.getSimulator()._goalConnections = 10000
env.getSimulator().setAllocator(first_fit_algorithm)
env.getSimulator().init()

#Start the simulator with goal connections times and without Agent interaction
env.start()
#Load the PPO2 Agent
model = PPO2(MlpPolicy, env, verbose=False)
#Start training in the simulator with 100 interation more, but with Agent interaction
model.learn(total_timesteps=100)
```

## Allocation function

The allocation function is the essential function for using this framework correctly. This function is called whenever one connection wants to start the connection.
The function receives from the environment seven parameters:
 
- src: The ID from the source node.
- dst: The ID from the destiny node.
- b: The random BitRate assigned by the Simulator.
- c: Connection Object
- n: Network Object
- path: Array list containing the possible connections routes.
- action: Answer from Agent 

In this example, the Agent decides which route (path) must be used to assign the network connection.

```python
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
        link = n.getLink(link._id)
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
```
## Documentation



## Acknowledgement

The following participants are greatly appreciated:
- [Gonzalo España](https://gitlab.com/GonzaloEspana)
- [Erick Viera]
- [Juan Pablo Sanchez]