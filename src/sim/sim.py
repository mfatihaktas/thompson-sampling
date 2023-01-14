import dataclasses

from src.env import bandit as bandit_module
from src.agent import agent as agent_module


@dataclasses.dataclass
class SimResult:
    reward_from_action_list = []
    reward_from_high_reward_list = []


def sim(
    bandit: bandit_module.Bandit,
    agent: agent_module.Agent,
    num_rounds: int,
) -> SimResult:
    sim_result = SimResult()

    for i in range(num_rounds):
        log(INFO, f">> i= {i}")

        arm_index = agent.next_action()
        reward_from_action = bandit.pull(arm_index=arm_index)
        reward_from_high_reward = bandit.pull_high_reward()
        log(DEBUG, "", arm_index=arm_index, sampled_reward=sampled_reward, reward_from_high_reward=reward_from_high_reward)

        sim_result.reward_from_action_list.append(reward_from_action)
        sim_result.reward_from_high_reward_list.append(reward_from_high_reward)

    return sim_result
