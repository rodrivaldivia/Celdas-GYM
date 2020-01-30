import random

# from pprint import pprint


class ReplayMemory():
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []

    def __str__(self):
        return str(self.__class__) + ": " + str(self.__dict__)

    def pushExperience(self, experience):
        self.memory.append(experience)
        if len(self.memory) > self.capacity:
            self.memory.pop(random.randint(0, len(self.memory) - 1))

    def sample(self, batchSize):
        if batchSize > len(self.memory):
            return random.sample(self.memory, len(self.memory))
        else:
            return random.sample(self.memory, batchSize)

    @property
    def numSamples(self):
        return len(self.memory)
