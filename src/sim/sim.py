from src.agent import agent as agent_module
from src.env import bandit as bandit_module
from src.sim import result as result_module
from src.utils.debug import *


def sim(
    bandit: bandit_module.Bandit,
    agent_list: list[agent_module.Agent],
    num_rounds: int,
) -> result_module.SimResult:
    sim_result = result_module.SimResult()

    for i in range(num_rounds):
        log(INFO, f">> i= {i}")

        for agent in agent_list:
            arm_index = agent.next_action()
            reward_from_action = bandit.pull(arm_index=arm_index)
            log(DEBUG, "", arm_index=arm_index, reward_from_action=reward_from_action)

            sim_result.append_reward_sample(agent=agent, reward=reward_from_action)

        reward_from_high_reward = bandit.pull_high_reward()
        log(DEBUG, "", reward_from_high_reward=reward_from_high_reward)

        sim_result.append_high_reward_sample(reward=reward_from_high_reward)

    return sim_result
