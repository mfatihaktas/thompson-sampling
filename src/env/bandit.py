import abc

from src.prob import rv
from src.env import arm as arm_module


class Bandit:
    def __init__(self, num_arms: int):
        self.num_arms = num_arms

    @abc.abstractmethod
    def pull(self, arm_index):
        pass


class StationaryBandit(Bandit):
    def __init__(
        self,
        num_arms: int,
        high_reward_rv: rv.RandomVariable,
        low_reward_rv: rv.RandomVariable,
    ):
        self.num_arms = num_arms
        self.high_reward_rv = high_reward_rv
        self.low_reward_rv = low_reward_rv

        self.arm_list = [
            arm_module.StationaryArm(
                name=f"high_reward_arm",
                reward_rv=self.high_reward_rv,
            )
        ]

        for i in range(num_arms - 1):
            self.arm_list.append(
                arm_module.StationaryArm(
                    name=f"low_reward_arm_{i}",
                    reward_rv=self.low_reward_rv,
                )
            )

    def __repr__(self):
        return (
            "StationaryBandit( \n"
            f"\t num_arms= {self.num_arms} \n"
            f"\t high_reward_rv= {self.high_reward_rv} \n"
            f"\t low_reward_rv= {self.low_reward_rv} \n"
            ")"
        )

    def pull(self, arm_index: int):
        return self.arm_list[arm_index].pull()
