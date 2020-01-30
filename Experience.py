
class Experience():
    def __init__(self, state, actionIndex, reward, nextState):
        self.state = state
        self.actionIndex = actionIndex
        self.reward = reward
        self.nextState = nextState

    def __str__(self):
        return str(self.__class__) + ": " + str(self.__dict__)
