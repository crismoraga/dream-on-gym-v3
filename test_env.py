import os
import gymnasium
import sys
sys.modules['gym'] = gymnasium
from dreamongymv2.gym_basic import *
from dreamongymv2.simNetPy import *

absolutepath = os.path.abspath(__file__)
fileDirectory = os.path.dirname(absolutepath)

env = gymnasium.make('rlonenv-v0', disable_env_checker=True)
# unwrap to access custom methods
raw_env = env.unwrapped

# define trivial reward and state
def reward():
    value = raw_env.getSimulator().lastConnectionIsAllocated()
    return 1 if value.name != Controller.Status.Not_Allocated.name else -1

def state():
    return 0

raw_env.setRewardFunc(reward)
raw_env.setStateFunc(state)
raw_env.initEnviroment(fileDirectory + '/NSFNet_4_bands.json', fileDirectory + '/routes.json')
raw_env.getSimulator().setGoalConnections(10)
raw_env.getSimulator().setAllocator(lambda *args, **kwargs: (Controller.Status.Not_Allocated, args[3]))
raw_env.getSimulator().init()
raw_env.start()
print('Environment started OK')
