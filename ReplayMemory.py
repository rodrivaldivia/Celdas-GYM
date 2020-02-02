import random
from collections import deque
from pprint import pprint

class ReplayMemory():
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)

    def __str__(self):
        return str(self.__class__) + ": " + str(self.__dict__)

    def pushExperience(self, experience):
        # print(experience)
        self.memory.append(experience)

    def popExperience(self):
        if (not len(self.memory)):
            return []
        return [self.memory.pop()]

    def sample(self, batchSize):
        if batchSize > len(self.memory):
            return random.sample(self.memory, len(self.memory))
        else:
            return random.sample(self.memory, batchSize)

    @property
    def numSamples(self):
        return len(self.memory)