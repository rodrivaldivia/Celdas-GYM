
import random

MIN_EPSILON = 0.01


class EpsilonStrategy():
    def __init__(self):
        self.epsilon = 1
        self.epsilonDecreaseRate = 0.0001

    def __str__(self):
        return str(self.__class__) + ": " + str(self.__dict__)

    def shouldExploit(self):
        if self.epsilon > MIN_EPSILON:
            self.epsilon -= self.epsilonDecreaseRate
        return random.uniform(0, 1) > self.epsilon

    def epsilon(self):
        return self.epsilon
