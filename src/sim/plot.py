from src.sim import (
    result as result_module,
    sim,
)
from src.utils.plot import *


def plot_mean_sim_result(mean_sim_result: result_module.MeanSimResult, title: str = None, plot_suffix: str = ""):
    num_rounds = len(mean_sim_result.mean_high_reward_list)
    round_index_list = list(range(1, num_rounds + 1))

    # Plot `mean_high_reward_list`
    plot.plot(round_index_list, mean_sim_result.mean_high_reward_list, color=next(dark_color_cycle), label="Highest", marker=next(marker_cycle), linestyle="dotted", lw=2, mew=3, ms=5)

    # Plot `agent_to_mean_rewards_map`
    for agent, mean_reward_list in mean_sim_result.agent_to_mean_rewards_map.items():
        plot.plot(round_index_list, mean_reward_list, color=next(dark_color_cycle), label=agent.name, marker=next(marker_cycle), linestyle="dotted", lw=2, mew=3, ms=5)

    fontsize = 14
    plot.legend(fontsize=fontsize)
    plot.ylabel("Mean reward", fontsize=fontsize)
    # plot.yscale("log")
    plot.xlabel("Round", fontsize=fontsize)

    if title:
        plot.title(title)

    # Save the plot
    plot.gcf().set_size_inches(10, 6)
    plot.savefig(f"plot_mean_sim_result_{plot_suffix}.png", bbox_inches="tight")
    plot.gcf().clear()
