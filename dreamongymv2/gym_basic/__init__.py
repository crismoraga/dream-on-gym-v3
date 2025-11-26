# -*- coding: utf-8 -*-
"""
Created on Wed Jun  1 17:33:53 2022

@author: redno
"""

from gymnasium.envs.registration import register 
register(id='rlonenv-v0',entry_point='dreamongymv2.gym_basic.envs:RlOnEnv',) 
