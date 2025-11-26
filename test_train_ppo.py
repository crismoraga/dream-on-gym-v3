import os
import sys
import gymnasium
sys.modules['gym'] = gymnasium
from dreamongymv2.simNetPy import *
from dreamongymv2.gym_basic import *
from stable_baselines3 import PPO

fileDirectory = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'examples', 'gym')

env = gymnasium.make('rlonenv-v0', disable_env_checker=True)
raw_env = env.unwrapped

def reward():
    value = raw_env.getSimulator().lastConnectionIsAllocated()
    return 1 if value.name != Controller.Status.Not_Allocated.name else -1

def state():
    return 0

raw_env.setRewardFunc(reward)
raw_env.setStateFunc(state)
raw_env.initEnviroment(os.path.join(fileDirectory, 'NSFNet_4_bands.json'), os.path.join(fileDirectory, 'routes.json'), '')
raw_env.getSimulator().setGoalConnections(1000)
raw_env.getSimulator().setMu(1)
raw_env.getSimulator().setLambda(1000)
raw_env.getSimulator().setAllocator(lambda *args, **kwargs: (Controller.Status.Not_Allocated, args[3]))
raw_env.getSimulator().init()
raw_env.start()

# small policy to test
model = PPO('MlpPolicy', env, verbose=0)
model.learn(total_timesteps=100)
print('PPO short training OK')
