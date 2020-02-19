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

import matplotlib.pyplot as plt

tf.compat.v1.enable_v2_behavior()


MEMORY_CAPACITY = 5000
MIN_REPLAY_MEMORY_SIZE = 1000
BATCH_SIZE = 60

NUM_ACTIONS = 5

STEPS_TO_UPDATE_NETWORK = 5
GAMMA = 0.8

directions = {
    'ACTION_DOWN':  (1,2),
    'ACTION_UP':    (1,0),
    'ACTION_RIGHT': (2,1),
    'ACTION_LEFT':  (0,1)
}


# 0 -> NONE
# 1 -> attack
# 2 -> left
# 3 -> right
# 4 -> down
# 5 -> up

# TODO chequear
availableActions = ['ACTION_USE', 'ACTION_LEFT', 'ACTION_RIGHT', 'ACTION_DOWN', 'ACTION_UP']

class Agent():
    def __init__(self):
        self.movementStrategy = EpsilonStrategy()
        self.replayMemory = ReplayMemory(MEMORY_CAPACITY)
        self.episode = 0
        self.policyNetwork = self._build_compile_model()
        self.targetNetwork = self._build_compile_model()
        if os.path.exists("./network/zelda-ddqn.h5"):
            print('Cargamos red')
            self.policyNetwork.load_weights("./network/zelda-ddqn.h5")
        print(self.policyNetwork.summary())
        self.exploreNext = False
        self.steps = 0
        self.averageLoss = 0
        self.averageReward = 0
        self.losses = []
        self.rewards = []

    """
    * Public method to be called at the start of every level of a game.
    * Perform any level-entry initialization here.
    """

    def init(self):   
        self.lastState = None
        self.lastAction = None
        self.align_target_model()
        self.foundKey = False
        self.switchedDirection = False
        self.currentDirection = 'ACTION_DOWN'
        self.keyPosition = None
        self.goalPosition = None

    def _build_compile_model(self):
        # inputs = Input(shape=(9,13), name='state')
        inputs = Input(shape=(3, 3, 3), name='state')
        x = Flatten()(inputs)
        # x = Dense(64, name='HiddenI', activation='relu')(x)
        x = Dense(100, name='HiddenII', activation='softmax')(x)
        outputs = Dense(NUM_ACTIONS, name='ActionsOutput', activation='relu')(x)

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

        if(self.steps % STEPS_TO_UPDATE_NETWORK == 0):    
            self.align_target_model()

        currentPosition = self.getAvatarCoordinates(state)

        if self.lastState is not None:
            # print(self.steps)
            reward = self.getReward(self.lastState, state, currentPosition)
            self.averageReward += reward
            # print(reward)
            self.replayMemory.pushExperience(Experience(self.lastState, self.lastAction, reward, state))
            if(self.steps % BATCH_SIZE == 0):
                loss = self.train()
                self.averageLoss += loss
        
        # pprint(vars(sso))
        # print(self.get_perception(sso))

        self.lastState = state
        self.steps += 1

        # print(state)

        tensorState = tf.convert_to_tensor([self.get_perception(state)])
        
        index = 0

        if self.movementStrategy.shouldExploit():

            # print('Using strategy...')
            q_values = self.policyNetwork.predict(tensorState)
            index = np.argmax(q_values[0])
            # print('q_values: ', q_values)
            # print(availableActions)
            # print('Predicted Best: ', availableActions[index])
            self.currentDirection = self.get_new_direction(availableActions[index])
            # print('Current direction: ', self.currentDirection)
            self.lastAction = np.argmax(q_values[0]) 
        else:
            index = random.randint(0, len(availableActions) -1)
            self.lastAction = index        
            # print(state)
            # print(self.get_perception(state))
            # print('Exploring: ', availableActions[index])
            self.currentDirection = self.get_new_direction(availableActions[index])
            # print('Current direction: ', self.currentDirection)
        # print(index+1)
        return index + 1

    def train(self):
        # print(self.replayMemory.numSamples)
        if self.replayMemory.numSamples < MIN_REPLAY_MEMORY_SIZE:
            return 0

        batch = self.replayMemory.sample(BATCH_SIZE)

        rawStates = [ self.get_perception(val.state) for val in batch ]

        states = tf.convert_to_tensor(rawStates, dtype=tf.float32)
        actions = np.array([val.actionIndex for val in batch])
        rewards = np.array([val.reward for val in batch])

        emptyState = np.ndarray((3,3))
        emptyState[:] = 0.0

        rawNextStates = [ emptyState if val.nextState is None else self.get_perception(val.state) for val in batch ] 

        nextStates = tf.convert_to_tensor(rawNextStates, dtype=tf.float32)

        # predict Q(s,a) given the batch of states
        prim_qt = self.policyNetwork(states)
        # predict Q(s',a') from the evaluation network
        prim_qtp1 = self.policyNetwork(nextStates)
        # copy the prim_qt into the target_q tensor - we then will update one index corresponding to the max action
        target_q = prim_qt.numpy()
        updates = rewards
        valid_idxs = np.array(nextStates).sum(axis=3).sum(axis=2).sum(axis=1) != 0
        batch_idxs = np.arange(BATCH_SIZE)
        # print('valid_idxs')
        # print(np.array(nextStates).sum(axis=3).sum(axis=2).sum(axis=1))
        # print(valid_idxs)
        # print(batch_idxs)

        prim_action_tp1 = np.argmax(prim_qtp1.numpy(), axis=1)
        q_from_target = self.targetNetwork(nextStates)
        # print(batch_idxs[valid_idxs])
        # print(q_from_target.numpy()[batch_idxs[valid_idxs]])

        updates[valid_idxs] += GAMMA * q_from_target.numpy()[batch_idxs[valid_idxs],
                                                             prim_action_tp1[valid_idxs]]
        target_q[batch_idxs, actions] = updates
        loss = self.policyNetwork.train_on_batch(states, target_q)
        return loss


        # batch = self.replayMemory.sample(BATCH_SIZE)
        # if len(batch) < BATCH_SIZE:
        #     return 0
        
        # # print('start training')

        # X = np.empty()
        # y = np.empty()

        # loss = 0

        # for experience in batch:

        #     tensorState = tf.convert_to_tensor([self.get_perception(experience.state)])
        #     tensorNextState = tf.convert_to_tensor([self.get_perception(experience.nextState)])

        #     # print(self.get_perception(experience.state))
        #     # Intentamos predecir la mejor accion
        #     target = self.policyNetwork.predict(tensorState)
        #     t = self.targetNetwork.predict(tensorNextState)
        #     # Para la accion que hicimos corregimos el Q-Value
        #     # print('Policy prediction: ', target)
        #     # print('Target prediction: ', t)
        #     # print('Q value before: ', target[0][experience.action])
        #     # print('Experience reward: ', experience.reward)
        #     # print('Target max: ', np.amax(t))
        #     target[0][experience.actionIndex] = experience.reward + GAMMA * np.amax(t)
        #     # print('Q value after: ', target[0][experience.action])
        #     # Entrenamos con la prediccion vs la correccion
        #     X.append(tensorState)
        #     y.append(target)
        #     # history = self.policyNetwork.fit(tensorState, target, verbose=0)
        #     # loss += history.history['loss'][0]

        # print(X.shape)
        # print(y.shape)
        # history = self.policyNetwork.fit(X, y, verbose=0)
        # return history.history['loss'][0]
        # self.policyNetwork.train_on_batch(X,y)
        # print('done training, loss: {}'.format(loss))
        # return loss


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

    def result(self, state, gameWinner):
        print("GAME OVER")
        self.gameOver = True
        if self.lastAction is not None:
            reward = 0
            if gameWinner == 'PLAYER_LOSES':
                reward = -50.0
                # print('AGENT KIA')
            elif gameWinner == 'PLAYER_WINS':
                reward = 50000.0
            self.replayMemory.pushExperience(Experience(self.lastState, self.lastAction, reward, state))
            loss = self.train()
            self.averageLoss += loss

        self.episode += 1

        if self.gameOver:
            avLoss = self.averageLoss/self.steps
            self.losses.append(avLoss)
            avReward = self.averageReward/self.steps
            self.rewards.append(avReward)

            print("Episode: {}, Reward: {}, avg loss: {}, eps: {}".format(
                self.episode, avReward, avLoss, self.movementStrategy.epsilon))
            print("Winner: {}".format(gameWinner))
            # with train_writer.as_default():
            #     tf.summary.scalar(
            #         'reward', self.averageReward, step=self.steps)
            #     tf.summary.scalar(
            #         'avg loss', self.averageLoss, step=self.steps)
        if self.episode % 10 == 0:
            self.policyNetwork.save_weights("./network/zelda-ddqn.h5")
            print('Model saved!')

    def plot(self):
        xLoss = range(len(self.losses))
        plt.plot(xLoss, self.losses, label='Loss')
        # xRewards = range(len(self.rewards))
        # plt.plot(xRewards, self.rewards, label='Reward')
        plt.xlabel('Episodios')
        # plt.ylabel('')
        plt.title("Resultados")
        plt.legend()
        plt.show()

    def getReward(self, lastState, currentState, currentPosition):
        level = lastState
        col = currentPosition[0] # col
        row = currentPosition[1] # row
        
        deltaDistance = self.getDistanceToGoal(self.getAvatarCoordinates(lastState)) - self.getDistanceToGoal(self.getAvatarCoordinates(currentState))
        reward = 1.0*(deltaDistance)

        moved = deltaDistance != 0

        # print(currentState.NPCPositionsNum)
        # print(self.getNumberOfEnemies(currentState))

        # if self.getNumberOfEnemies(currentState) < self.getNumberOfEnemies(lastState):
        if self.getNumberOfEnemies(currentState) < self.getNumberOfEnemies(lastState):
            print('KILLED AN ENEMY')
            return 500.0

        # Found key
        if level[col][row] == 2.0:
            print('FOUND KEY')
            self.foundKey = True
            return 10000.0

        if not moved:
            # print('DID NOT MOVE')
            if self.lastAction is None:
                # print('LAST ACCION IS NONE')
                return -10.0
            if 0 < self.lastAction < NUM_ACTIONS and availableActions[self.lastAction] == 'ACTION_USE':
                # print('BUT DID ATTACK')
                reward = -5.0
            else:
                if(self.switchedDirection):
                    # print('SWITCHED DIRECTION')
                    reward = -10.0
                else:
                    # print ('STEPPED INTO WALL')
                    reward = -20.0
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
        avatar = self.getElementCoordinates(state, 1.0)
        if avatar is not None: 
            return self.getElementCoordinates(state, 1.0)
        else:
            return 

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

        if avatarPosition is not None:
            for ii in range(3):                   
                for jj in range(3):
                    iiPosition = ii + avatarPosition[0] - 1
                    jjPosition = jj + avatarPosition[1] - 1
                    # print([jjPosition, iiPosition])
                    # print(self.getDistanceToGoal([int(jjPosition), int(iiPosition)]))
                    distances[jj][ii] = self.getDistanceToGoal([int(jjPosition), int(iiPosition)])
                    level[jj][ii] = state[iiPosition][jjPosition]
            # print(level)
            # print(distances)
        return [level, distances, direction]
