import abc

from src.env import arm as arm_module
from src.prob import rv
from src.utils.debug import *


class Bandit:
    def __init__(self, num_arms: int):
        self.num_arms = num_arms

    @abc.abstractmethod
    def pull(self, arm_id):
        pass

    @abc.abstractmethod
    def pull_high_reward(self):
        pass

    @abc.abstractmethod
    def mean_high_reward(self):
        pass


class StationaryBandit(Bandit):
    def __init__(
        self,
        num_arms: int,
        num_arms_w_high_reward: int,
        high_reward_rv: rv.RandomVariable,
        low_reward_rv: rv.RandomVariable,
    ):
        self.num_arms = num_arms
        self.num_arms_w_high_reward = num_arms_w_high_reward
        self.high_reward_rv = high_reward_rv
        self.low_reward_rv = low_reward_rv

        check(self.num_arms_w_high_reward <= self.num_arms, "")

        self.arm_list = [
            arm_module.StationaryArm(
                name=f"high_reward_arm_{i}",
                reward_rv=self.high_reward_rv,
            )
            for i in range(self.num_arms_w_high_reward)
        ]

        for i in range(self.num_arms - self.num_arms_w_high_reward):
            self.arm_list.append(
                arm_module.StationaryArm(
                    name=f"low_reward_arm_{i}",
                    reward_rv=self.low_reward_rv,
                )
            )

        log(DEBUG, "Constructed", arm_list=self.arm_list)

    def __repr__(self):
        return (
            "StationaryBandit( \n"
            f"\t num_arms= {self.num_arms} \n"
            f"\t num_arms_w_high_reward= {self.num_arms_w_high_reward} \n"
            f"\t high_reward_rv= {self.high_reward_rv} \n"
            f"\t low_reward_rv= {self.low_reward_rv} \n"
            ")"
        )

    def pull(self, arm_id: int):
        return self.arm_list[arm_id].pull()

    def pull_high_reward(self):
        # Note: High-reward arms are at the beginning of `arm_list`.
        return self.high_reward_rv.sample()

    def mean_high_reward(self):
        return self.high_reward_rv.mean()


class NonStationaryBandit(Bandit):
    def __init__(
        self,
        num_arms: int,
        num_arms_w_high_reward: int,
        num_arms_w_medium_reward: int,
        high_reward_rv: rv.RandomVariable,
        medium_reward_rv: rv.RandomVariable,
        low_reward_rv: rv.RandomVariable,
        phase_duration_rv: rv.RandomVariable,
    ):
        self.num_arms_w_high_reward = num_arms_w_high_reward
        self.num_arms_w_medium_reward = num_arms_w_medium_reward
        self.high_reward_rv = high_reward_rv
        self.medium_reward_rv = medium_reward_rv
        self.low_reward_rv = low_reward_rv
        self.phase_duration_rv = phase_duration_rv

        self.num_arms_w_low_reward = num_arms - self.num_arms_w_high_reward - self.num_arms_w_medium_reward
        check(self.num_arms_w_low_reward >= 0, "")

        self.arm_list = [
            arm_module.NonStationaryArm_wHighLowReward(
                name=f"non-stationary_high/low_reward_arm_{i}",
                high_reward_rv=self.high_reward_rv,
                low_reward_rv=self.low_reward_rv,
                phase_duration_rv=self.phase_duration_rv,
            )
            for i in range(self.num_arms_w_high_reward)
        ]

        for i in range(self.num_arms_w_medium_reward):
            self.arm_list.append(
                arm_module.StationaryArm(
                    name=f"medium_reward_arm_{i}",
                    reward_rv=self.medium_reward_rv,
                )
            )

        for i in range(self.num_arms_w_medium_reward):
            self.arm_list.append(
                arm_module.StationaryArm(
                    name=f"low_reward_arm_{i}",
                    reward_rv=self.low_reward_rv,
                )
            )

        log(DEBUG, "Constructed", arm_list=self.arm_list)

    def __repr__(self):
        return (
            "NonStationaryBandit( \n"
            f"\t num_arms_w_high_reward= {self.num_arms_w_high_reward} \n"
            f"\t num_arms_w_medium_reward= {self.num_arms_w_medium_reward} \n"
            f"\t num_arms_w_low_reward= {self.num_arms_w_low_reward} \n"
            f"\t high_reward_rv= {self.high_reward_rv} \n"
            f"\t low_reward_rv= {self.low_reward_rv} \n"
            f"\t phase_duration_rv= {self.phase_duration_rv} \n"
            ")"
        )

    def pull(self, arm_id: int):
        return self.arm_list[arm_id].pull()

    def pull_high_reward(self):
        # Note: High-reward arms are at the beginning of `arm_list`.
        return self.high_reward_rv.sample()

    def mean_high_reward(self):
        return self.high_reward_rv.mean()
