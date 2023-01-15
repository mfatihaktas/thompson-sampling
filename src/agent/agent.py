import abc
import collections

from typing import Tuple

from src.prob import rv
from src.utils.debug import *


class Agent(abc.ABC):
    def __init__(self, num_arms: int):
        self.num_arms = num_arms

    def __repr__(self):
        return (
            "Agent( \n"
            f"\t num_arms= {self.num_arms} \n"
            ")"
        )

    @abc.abstractmethod
    def next_action(self):
        pass

    @abc.abstractmethod
    def observe(self, arm_index: int, reward: float):
        pass


class ThompsonSamplingAgent(Agent):
    def __init__(self, num_arms: int, win_len: int):
        super().__init__(num_arms=num_arms)

        self.win_len = win_len

        self.arm_id_to_reward_queue_map = {
            arm_id: collections.deque(maxlen=win_len) for arm_id in range(self.num_arms)
        }

    def mean_stdev_reward(self, arm_id: str) -> Tuple[float, float]:
        reward_queue = self.arm_id_to_reward_queue_map[arm_id]
        mean = numpy.mean(reward_queue) if len(reward_queue) else 0
        stdev = numpy.std(reward_queue) if len(reward_queue) else 1
        check(stdev >= 0, "Stdev cannot be negative")
        if stdev == 0:
            stdev = 1

        return mean, stdev


class ThompsonSamplingAgent_slidingWin(ThompsonSamplingAgent):
    def __init__(self, num_arms: int, win_len: int):
        super().__init__(num_arms=num_arms, win_len=win_len)

    def __repr__(self):
        return (
            "ThompsonSamplingAgent_slidingWin( \n"
            f"\t num_arms= {self.num_arms} \n"
            f"\t win_len= {self.win_len} \n"
            ")"
        )

    def next_action(self) -> int:
        log(DEBUG, "", arm_id_to_reward_queue_map=self.arm_id_to_reward_queue_map)

        # Choose the node with min wait time sample
        action, max_sample = None, float("-Inf")
        for arm_id in range(self.num_arms):
            mean, stdev = self.mean_stdev_reward(arm_id=arm_id)

            # s = rv.TruncatedNormal(mu=mean, sigma=stdev).sample()
            s = rv.Normal(mu=mean, sigma=stdev).sample()
            if s > max_sample:
                max_sample = s
                action = arm_id
                # log(DEBUG, "s > max_sample", s=s, max_sample=max_sample, action=action)

        return action

    def observe(self, arm_index: int, reward: float):
        self.arm_id_to_reward_queue_map[arm_id].append(reward)
        log(DEBUG, "recorded", arm_id=arm_id, reward=reward)


class ThompsonSamplingAgent_resetWinOnRareEvent(ThompsonSamplingAgent):
    def __init__(self, num_arms: int, win_len: int, tail_mass_threshold: float):
        super().__init__(num_arms=num_arms, win_len=win_len)

        self.tail_mass_threshold = tail_mass_threshold

    def __repr__(self):
        return (
            "ThompsonSamplingAgent_resetWinOnRareEvent( \n"
            f"\t num_arms= {self.num_arms} \n"
            f"\t win_len= {self.win_len} \n"
            ")"
        )

    def next_action(self) -> int:
        log(DEBUG, "", arm_id_to_reward_queue_map=self.arm_id_to_reward_queue_map)

        # Choose the node with min wait time sample
        action, max_sample = None, float("-Inf")
        for arm_id in range(self.num_arms):
            mean, stdev = self.mean_stdev_reward(arm_id=arm_id)

            # s = rv.TruncatedNormal(mu=mean, sigma=stdev).sample()
            s = rv.Normal(mu=mean, sigma=stdev).sample()
            if s > max_sample:
                max_sample = s
                action = arm_id
                # log(DEBUG, "s > max_sample", s=s, max_sample=max_sample, action=action)

        return action

    def observe(self, arm_index: int, reward: float):
        def record():
            self.arm_id_to_reward_queue_map[arm_id].append(reward)
            log(DEBUG, "recorded", arm_id=arm_id, reward=reward)

        if len(self.arm_id_to_reward_queue_map[node_id]) < 5:
            record()

        else:
            mean, stdev = self.mean_stdev_reward(node_id)
            # reward_rv = rv.TruncatedNormal(mu=mean, sigma=stdev)
            reward_rv = rv.Normal(mu=mean, sigma=stdev)

            Pr_getting_larger_than_reward = reward_rv.tail_prob(reward)
            Pr_getting_smaller_than_reward = reward_rv.cdf(reward)
            tail_mass = min(Pr_getting_larger_than_reward, Pr_getting_smaller_than_reward)
            if tail_mass <= self.tail_mass_threshold:
                log(DEBUG, "Rare event detected", reward=reward, mean=mean, stdev=stdev, tail_mass=tail_mass, tail_mass_threshold=self.tail_mass_threshold)
                self.arm_id_to_reward_queue_map[node_id].clear()
            else:
                record()
