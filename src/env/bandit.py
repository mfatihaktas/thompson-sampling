import abc


class Bandit:
    def __init__(self, num_arms: int):
        self.num_arms = num_arms

    @abc.abstractmethod
    def pull_arm(self, arm_index):
        pass


class StationaryBandit(Bandit):
    def __init__(self, num_arms: int):
        self.num_arms = num_arms

    def pull_arm(self, arm_index: int):
        pass
