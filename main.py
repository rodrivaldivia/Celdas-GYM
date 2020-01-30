import gym
import gym_gvgai
import numpy as np
import tensorflow as tf
import datetime as dt
import os
from random import randint
from Agent import Agent

tf.compat.v1.enable_v2_behavior()

num_episodes = 100  # @param {type:"integer"}
num_steps = 1000  # @param {type:"integer"}
render = True  # @param {type:"boolean"}

STORE_PATH = os.getcwd()


env_name = 'gvgai-zelda-lvl0-v0'
env = gym_gvgai.make(env_name)

print('Starting ' + env.env.game + " with Level " + str(env.env.lvl))


def grayToArray(array):
	result = np.zeros((9, 13))
	for i in range(int(array.shape[0]/10)):
		for j in range(int(array.shape[1]/10)):
			result[i][j] = int(array[10*i+5, 10*j+5])
			elem = result[i][j]
			if elem == 53:  # Empty
				result[i][j] = 0.0
			if elem == 201 or elem == 38:  # Avatar
				result[i][j] = 1.0
			if elem == 123:  # Key
				result[i][j] = 2.0
			if elem == 52:  # Door
				result[i][j] = 3.0
			if elem == 61:  # Enemy
				result[i][j] = 4.0
			if elem == 127 or elem == 92:  # Wall
				result[i][j] = 5.0

	return result

def getState():
    rgb = env.render('rgb_array')
    gray = np.mean(rgb, -1)
    return grayToArray(gray)

agent = Agent()

for i in range(num_episodes):  # testing 100 times
    current_score = 0  # record current testing round score
    env.reset()
    state = getState()
    agent.init()
    for step in range(num_steps):
        if render:
          env.render()
        action = agent.act(state)
        stateObs, increScore, done, debug = env.step(action)
        state = getState()
        if done:
            agent.result(state, debug['winner'])
            print("Game over at game tick " + str(step+1) + " with player " + debug['winner'])
            break
