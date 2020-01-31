import random
import os
import datetime as dt
from time import sleep

from EpsilonStrategy import EpsilonStrategy
from ReplayMemory import ReplayMemory
from Experience import Experience

import math
import numpy as np
from pprint import pprint
from scipy.spatial import distance

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Dense, Input, Reshape, Flatten

tf.compat.v1.enable_v2_behavior()

# tf.reset_default_graph()


MEMORY_CAPACITY = 1000
# TIMESTEPS_PER_EPISODE = 150
STEPS_TO_UPDATE_NETWORK = 3
MIN_REPLAY_MEMORY_SIZE = 500
NUM_ACTIONS = 5
BATCH_SIZE = 20
GAMMA = 0.8

directions = {
    'ACTION_DOWN':  (1,2),
    'ACTION_UP':    (1,0),
    'ACTION_RIGHT': (2,1),
    'ACTION_LEFT':  (0,1)
}

# TODO chequear
availableActions = ['ACTION_USE', 'ACTION_UP', 'ACTION_LEFT', 'ACTION_RIGHT', 'ACTION_DOWN']

class Agent():
    def __init__(self):
        self.movementStrategy = EpsilonStrategy()
        self.replayMemory = ReplayMemory(MEMORY_CAPACITY)
        self.episode = 0
        self.policyNetwork = self._build_compile_model()
        self.targetNetwork = self._build_compile_model()
        if self.episode == 0 and os.path.exists("./celdas/network/zelda.index"):
            self.policyNetwork.load_weights("./celdas/network/zelda")
        print(self.policyNetwork.summary())
        self.exploreNext = False

    """
    * Public method to be called at the start of every level of a game.
    * Perform any level-entry initialization here.
    """

    def init(self):   
        self.lastState = None
        self.lastAction = None
        self.steps = 0
        self.align_target_model()
        self.foundKey = False
        self.switchedDirection = False
        self.currentDirection = 'ACTION_DOWN'
        self.keyPosition = None
        self.goalPosition = None
        self.averageLoss = 0
        self.averageReward = 0

    def _build_compile_model(self):
        # inputs = Input(shape=(9,13), name='state')
        inputs = Input(shape=(3, 3, 3), name='state')
        x = Flatten()(inputs)
        x = Dense(30, activation='relu')(x)
        x = Dense(64, activation='relu')(x)
        outputs = Dense(NUM_ACTIONS, activation='softmax')(x)

        model = Model(inputs=inputs, outputs=outputs, name='Zelda')

        model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(learning_rate=0.001))
        return model

    def align_target_model(self):
        self.targetNetwork.set_weights(self.policyNetwork.get_weights())
   

    def get_new_direction(self, action):
        # print('New action: ', action)
        # print('Old action: ',self.currentDirection)
        # print(action == 'ACTION_USE')
        self.switchedDirection = action != self.currentDirection and action != 'ACTION_USE'
        # print(self.switchedDirection)
        return self.currentDirection if action == 'ACTION_USE' else action 

    """
     * Method used to determine the next move to be performed by the agent.
     * This method can be used to identify the current state of the game and all
     * relevant details, then to choose the desired course of action.
     *
     * @param sso Observation of the current state of the game to be used in deciding
     *            the next action to be taken by the agent.
     * @param elapsedTimer Timer (40ms)
     * @return The action to be performed by the agent.
     """

    def act(self, state):

        if self.keyPosition is None:
            self.keyPosition = self.getKeyPosition(state)
            self.exitPosition = self.getExitPosition(state)


        # if(sso.gameTick%BATCH_SIZE==0):
            # print('train')
            # self.train()
        if(self.steps % STEPS_TO_UPDATE_NETWORK == 0):    
            self.align_target_model()

        currentPosition = self.getAvatarCoordinates(state)

        if self.lastState is not None:
            reward = self.getReward(self.lastState, state, currentPosition)
            # print(reward)
            self.replayMemory.pushExperience(Experience(self.lastState, self.lastAction, reward, state))
        
        # pprint(vars(sso))
        # print(self.get_perception(sso))

        self.lastState = state
        self.steps += 1

        tensorState = tf.convert_to_tensor([self.get_perception(state)])

        if self.movementStrategy.shouldExploit() and not self.exploreNext:

            # print('Using strategy...')
            q_values = self.policyNetwork.predict(tensorState)
            index = np.argmax(q_values[0])
            print('q_values: ', q_values)
            print(availableActions)
            print('Predicted Best: ', availableActions[index])
            self.currentDirection = self.get_new_direction(availableActions[index])
            print('Current direction: ', self.currentDirection)
            return np.argmax(q_values[0])
        else:
            self.exploreNext = False
            # print('Exploring...')
            if self.steps == 3000:
                return "ACTION_ESCAPE"
            else:
                index = random.randint(0, len(availableActions) - 1)
                self.lastAction = index        
                print(state)
                print(self.get_perception(state))
                print('Exploring: ', availableActions[index])
                self.currentDirection = self.get_new_direction(availableActions[index])
                # print('Current direction: ', self.currentDirection)
                return index

    def train(self):
        # print(self.replayMemory.numSamples)
        if self.replayMemory.numSamples < MIN_REPLAY_MEMORY_SIZE:
            return
        batch = self.replayMemory.sample(BATCH_SIZE)
        if len(batch) < BATCH_SIZE:
            return
        
        print('start training')

        # X = []
        # y = []

        loss = 0

        for experience in batch:

            tensorState = tf.convert_to_tensor([self.get_perception(experience.state)])
            tensorNextState = tf.convert_to_tensor([self.get_perception(experience.nextState)])

            # print(self.get_perception(experience.state))
            # Intentamos predecir la mejor accion
            target = self.policyNetwork.predict(tensorState)
            t = self.targetNetwork.predict(tensorNextState)
            # Para la accion que hicimos corregimos el Q-Value
            # print('Policy prediction: ', target)
            # print('Target prediction: ', t)
            # print('Q value before: ', target[0][experience.action])
            # print('Experience reward: ', experience.reward)
            # print('Target max: ', np.amax(t))
            target[0][experience.action] = experience.reward + GAMMA * np.amax(t)
            # print('Q value after: ', target[0][experience.action])

            # Entrenamos con la prediccion vs la correccion
            X.append(tensorState)
            # y.append(target)
            loss += self.policyNetwork.fit(tensorState, target, verbose=0)
        # self.policyNetwork.train_on_batch(X,y)
        print('done training')
        return loss


    """
    * Method used to perform actions in case of a game end.
    * This is the last thing called when a level is played (the game is already in a terminal state).
    * Use this for actions such as teardown or process data.
    *
    * @param sso The current state observation of the game.
    * @param elapsedTimer Timer (up to CompetitionParameters.TOTAL_LEARNING_TIME
    * or CompetitionParameters.EXTRA_LEARNING_TIME if current global time is beyond TOTAL_LEARNING_TIME)
    * @return The next level of the current game to be played.
    * The level is bound in the range of [0,2]. If the input is any different, then the level
    * chosen will be ignored, and the game will play a random one instead.
    """

    def result(self, sso, gameWinner):
        print("GAME OVER")
        self.gameOver = True
        self.episode += 1
        self.policyNetwork.save_weights("./celdas/network/zelda")
        print('Model saved!')
        if self.lastAction is not None:
            if gameWinner == 'PLAYER_LOSES':
                reward += -100.0
                print('AGENT KIA')
            elif gameWinner == 'PLAYER_WINS':
                reward += 10000.0
            self.replayMemory.pushExperience(Experience(self.lastState, self.lastAction, reward, sso))
            loss = self.train()
            self.averageLoss += loss
            self.averageReward += reward

        # self.episode += 1

        # if self.gameOver:
        #     self.averageLoss /= self.steps
        #     print("Episode: {}, Reward: {}, avg loss: {}, eps: {}".format(
        #         self.episode, self.averageReward, self.averageLoss, self.movementStrategy.epsilon))
        #     print("Winner: {}".format(gameWinner))
        #     with train_writer.as_default():
        #         tf.summary.scalar(
        #             'reward', self.averageReward, step=self.steps)
        #         tf.summary.scalar(
        #             'avg loss', self.averageLoss, step=self.steps)
        # if self.episode % 10 == 0:
        #     self.policyNetwork.save_weights("./network/zelda-ddqn.h5")
        #     print('Model saved!')


    def getReward(self, lastState, currentState, currentPosition):
        level = lastState
        col = currentPosition[0] # col
        row = currentPosition[1] # row
        
        deltaDistance = self.getDistanceToGoal(self.getAvatarCoordinates(lastState)) - self.getDistanceToGoal(self.getAvatarCoordinates(currentState))
        reward = 10.0*(deltaDistance)

        moved = deltaDistance != 0

        # print(currentState.NPCPositionsNum)
        # print(self.getNumberOfEnemies(currentState))

        # if self.getNumberOfEnemies(currentState) < self.getNumberOfEnemies(lastState):
        if self.getNumberOfEnemies(currentState) < self.getNumberOfEnemies(lastState):
            print('KILLED AN ENEMY')
            return 500.0

        if not moved:
            print('DID NOT MOVE')
            # print(currentState.availableActions[self.lastAction])
            if 0 < self.lastAction < NUM_ACTIONS and availableActions[self.lastAction] == 'ACTION_USE':
                print('BUT DID ATTACK')
                reward = -10.0
            else:
                if(self.switchedDirection):
                    print('SWITCHED DIRECTION')
                    reward = 0.0
                else:
                    print ('STEPPED INTO WALL')
                    reward = -50.0
        # elif level[col][row] == elementToFloat['.']:
            # print ('MOVED')
            # print (self.getDistanceToGoal(currentState))
        elif level[col][row] == 3.0:
            print ('FOUND KEY')
            # Found key
            self.foundKey = True
            # Set GATE as new goal
            # self.goalPosition = currentState.portalsPositions[0][0].getPositionAsArray()
            reward = 1000.0
        elif level[col][row] == 2.0 and self.foundKey:
            # Won
            print('WON')
            reward = 5000.0
        # else:
        #     print ('No entro a nignuno')

        # print 'level: '
        # print level[col][row]
        print(reward)
        return reward
    
    def getElementCoordinates(self, state, element):
        result = None
        for i in range(len(state)):
            for j in range(len(state[0])):
                if state[i][j] == element:
                    result = [i, j]
        return result

    def getNumberOfEnemies(self, state):
        result = 0
        for i in range(len(state)):
            for j in range(len(state[0])):
                if state[i][j] == 4.0:
                    result += 1
        return result

    def getAvatarCoordinates(self, state):
        return self.getElementCoordinates(state, 1.0)

    def getKeyPosition(self, state):
        return self.getElementCoordinates(state, 2.0)

    def getExitPosition(self, state):
        return self.getElementCoordinates(state, 3.0)

    def getDistanceToKey(self, state):
        return distance.cityblock(self.getAvatarCoordinates(state), self.keyPosition)

    def getDistanceToExit(self, state):
        return distance.cityblock(self.getAvatarCoordinates(state), self.exitPosition)

    def getDistanceToGoal(self, coordinates):
        # print('Getting distance to goal')
        # print(self.getAvatarCoordinates(state))
        # print(self.goalPosition)
        goalPosition = self.keyPosition
        if(self.foundKey):
            goalPosition = self.exitPosition
        return distance.cityblock(coordinates, goalPosition)

    def isCloserToKey(self, previousState, currentState):
        return self.getDistanceToKey(currentState) < self.getDistanceToKey(previousState)

    def isCloserToExit(self, previousState, currentState):
        return self.getDistanceToExit(currentState) < self.getDistanceToExit(previousState)


    def get_perception(self, state):
        avatarPosition = self.getAvatarCoordinates(state)
        level = np.ndarray((3,3))
        distances = np.ndarray((3,3))
        direction = np.ndarray((3,3))

        level[:] = 0.0
        distances[:] = 20
        direction[:] = 0.0
        direction[directions[self.currentDirection]] = 1.0

        # level[:] = '.'
        for ii in range(3):                   
            for jj in range(3):
                iiPosition = ii + avatarPosition[1] - 1 
                jjPosition = jj + avatarPosition[0] - 1 
                # print([jjPosition, iiPosition])
                # print(self.getDistanceToGoal([int(jjPosition), int(iiPosition)]))
                distances[jj][ii] = self.getDistanceToGoal([int(jjPosition), int(iiPosition)])
                level[jj][ii] = state[iiPosition][jjPosition]
        print(level)
        print(distances)
        return [level, distances, direction]
