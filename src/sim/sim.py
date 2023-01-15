import collections
import dataclasses

from src.env import bandit as bandit_module
from src.agent import agent as agent_module


@dataclasses.dataclass
class SimResult:
    agent_to_rewards_from_action_map: dict[agent_module.Agent, list] = dataclasses.field(
        # default=collections.defaultdict(list)
        default_factory=collections.defaultdict(list)
    )
    reward_from_high_reward_list: list = dataclasses.field(
        default_factory=list
    )

    def append_reward_from_action(self, agent: agent_module.Agent, reward: float):
        self.sim_result.agent_to_rewards_from_action_map[agent].append(reward)

    def append_reward_from_high_reward(self, reward: float):
        self.reward_from_high_reward_list.append(reward)


def sim(
    bandit: bandit_module.Bandit,
    agent_list: list[agent_module.Agent],
    num_rounds: int,
) -> SimResult:
    sim_result = SimResult()

    for i in range(num_rounds):
        log(INFO, f">> i= {i}")

        for agent in agent_list:
            arm_index = agent.next_action()
            reward_from_action = bandit.pull(arm_index=arm_index)
            log(DEBUG, "", arm_index=arm_index, sampled_reward=sampled_reward)

            sim_result.append_reward_from_action(agent=agent, reward=reward_from_action)

        reward_from_high_reward = bandit.pull_high_reward()
        log(DEBUG, "", reward_from_high_reward=reward_from_high_reward)

        sim_result.append_reward_from_high_reward(reward=reward_from_high_reward)

    return sim_result
