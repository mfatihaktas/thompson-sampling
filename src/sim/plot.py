from src.sim import sim
from src.utils.plot import *


def plot_sim_result(sim_result: sim.SimResult):
    num_rounds = len(sim_result.high_reward_sample_list)
    round_index_list = list(range(1, num_rounds + 1))

    sim_result.agent_to_reward_samples_map

    plot.plot(round_index_list, sim_result.high_reward_sample_list, color=next(dark_color_cycle), label="Highest", marker=next(marker_cycle), linestyle="dotted", lw=2, mew=3, ms=5)
