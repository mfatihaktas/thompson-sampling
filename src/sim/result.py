import collections
import dataclasses

from src.agent import agent as agent_module


@dataclasses.dataclass
class SimResult:
    agent_to_reward_samples_map: dict[agent_module.Agent, list] = dataclasses.field(
        default_factory=lambda: collections.defaultdict(list)
    )
    high_reward_sample_list: list = dataclasses.field(
        default_factory=list
    )

    def append_reward_sample(self, agent: agent_module.Agent, reward: float):
        self.agent_to_reward_samples_map[agent].append(reward)

    def append_high_reward_sample(self, reward: float):
        self.high_reward_sample_list.append(reward)


class MeanSimResult:
    def __init__(self, sim_result_list: list[SimResult]):
        # Construct `agent_to_reward_samples_list_map`
        agent_to_reward_samples_list_map = collections.defaultdict(list)
        for sim_result in sim_result_list:
            for agent, reward_list in sim_result.agent_to_reward_samples_map.items():
                agent_to_reward_samples_list_map[agent].append(reward_list)

        self.agent_to_mean_rewards_map = {
            agent: self.get_mean_rewards(reward_samples_list=reward_samples_list)
            for agent, reward_samples_list in agent_to_reward_samples_list_map.items()
        }

        # Construct `mean_high_reward_list`
        self.mean_high_reward_list = self.get_mean_rewards(
            reward_samples_list=[sim_result.high_reward_sample_list for sim_result in sim_result_list]
        )

    def __repr__(self):
        return (
            "MeanSimResult( \n"
            f"agent_to_mean_rewards_map= \n{self.agent_to_mean_rewards_map} \n"
            f"mean_high_reward_list= \n{self.mean_high_reward_list} \n"
            ")"
        )

    def get_mean_rewards(self, reward_samples_list: list[list[float]]) -> list[float]:
        if len(reward_samples_list) == 0:
            return []

        mean_reward_list = reward_samples_list[0][:]
        count = 1

        for i, reward_list in enumerate(reward_samples_list):
            if i == 0:
                continue

            for j, reward in enumerate(reward_list):
                mean_reward_list[j] = mean_reward_list[j]*count / (count + 1) + reward / (count + 1)

            count += 1

        return mean_reward_list
