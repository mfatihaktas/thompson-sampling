from src.agent import agent as agent_module
from src.env import bandit as bandit_module
from src.sim import result as result_module
from src.utils.debug import *


def sim_single_run(
    bandit: bandit_module.Bandit,
    agent_list: list[agent_module.Agent],
    num_rounds: int,
) -> result_module.SimResult:
    sim_result = result_module.SimResult()

    for i in range(num_rounds):
        log(INFO, f">> i= {i}")

        for agent in agent_list:
            arm_id = agent.next_action()
            reward_sample = bandit.pull(arm_id=arm_id)
            log(DEBUG, "", arm_id=arm_id, reward_sample=reward_sample)

            sim_result.append_reward_sample(agent=agent, reward=reward_sample)
            agent.observe(arm_id=arm_id, reward=reward_sample)

        sample_from_high_reward = bandit.pull_high_reward()
        log(DEBUG, "", sample_from_high_reward=sample_from_high_reward)

        sim_result.append_high_reward_sample(reward=sample_from_high_reward)

    return sim_result


def sim(
    bandit: bandit_module.Bandit,
    agent_list: list[agent_module.Agent],
    num_rounds: int,
    num_sim_runs: int = 1,
) -> result_module.MeanSimResult:
    log(INFO, "Started", bandit=bandit, agent_list=agent_list, num_rounds=num_rounds, num_sim_runs=num_sim_runs)

    sim_result_list = []
    for i in range(num_sim_runs):
        sim_result = sim_single_run(
            bandit=bandit,
            agent_list=agent_list,
            num_rounds=num_rounds,
        )
        sim_result_list.append(sim_result)

    mean_sim_result = result_module.MeanSimResult(sim_result_list=sim_result_list)
    log(INFO, "Done")
    return mean_sim_result
