import pytest

from typing import Tuple

from src.agent import agent as agent_module
from src.env import bandit as bandit_module
from src.prob import rv
from src.sim import plot, sim
from src.utils.debug import *


@pytest.fixture(
    params=[
        (3, 1, 1),
    ]
)
def num_arms_and_w_high_and_medium_reward(request) -> Tuple[int, int, int]:
    return request.param


@pytest.fixture(
    params=[
        (rv.Normal(mu=10, sigma=1), rv.Normal(mu=5, sigma=1), rv.Normal(mu=1, sigma=1)),
    ]
)
def high_and_medium_and_low_reward_rv(request) -> Tuple[rv.RandomVariable, rv.RandomVariable, rv.RandomVariable]:
    return request.param

@pytest.fixture(
    params=[
        rv.CustomDiscrete(
            value_list=[50],
            prob_weight_list=[1],
        ),
    ]
)
def phase_duration_rv(request) -> rv.RandomVariable:
    return request.param


# @pytest.mark.parametrize(
#     "num_arms",
#     (2, 3),
#     indirect=True
# )
def test_stationary_bandit_w_ts_sliding_vs_reset_win(
    num_arms_and_w_high_and_medium_reward: int,
    high_and_medium_and_low_reward_rv: Tuple[rv.RandomVariable, rv.RandomVariable, rv.RandomVariable],
    phase_duration_rv: rv.RandomVariable,
):
    log(INFO, "Started",
        num_arms_and_w_high_and_medium_reward=num_arms_and_w_high_and_medium_reward,
        high_and_medium_and_low_reward_rv=high_and_medium_and_low_reward_rv,
        phase_duration_rv=phase_duration_rv,
    )

    num_arms, num_arms_w_high_reward, num_arms_w_medium_reward = num_arms_and_w_high_and_medium_reward
    high_reward_rv, _, low_reward_rv = high_and_medium_and_low_reward_rv

    bandit = bandit_module.StationaryBandit(
        num_arms=num_arms,
        num_arms_w_high_reward=num_arms_w_high_reward,
        high_reward_rv=high_reward_rv,
        low_reward_rv=low_reward_rv,
    )

    agent_ts = agent_module.ThompsonSamplingAgent_full(
        name="TS",
        num_arms=num_arms,
    )

    win_len = 100
    agent_ts_sliding_win = agent_module.ThompsonSamplingAgent_slidingWin(
        name="TS-SlidingWin",
        num_arms=num_arms,
        win_len=win_len,
    )

    agent_ts_reset_win = agent_module.ThompsonSamplingAgent_resetWinOnRareEvent(
        name="TS-ResetWin",
        num_arms=num_arms,
        win_len=win_len,
        tail_mass_threshold=0.05,
    )

    num_rounds = 1000
    num_sim_runs = 2
    mean_sim_result = sim.sim(
        bandit=bandit,
        # agent_list=[agent_ts],
        # agent_list=[agent_ts, agent_ts_sliding_win],
        agent_list=[agent_ts, agent_ts_sliding_win, agent_ts_reset_win],
        num_rounds=num_rounds,
        num_sim_runs=num_sim_runs,
    )

    title = r"$N_{\textrm{arm}}=$" + f"{num_arms}, " + \
        r"$N_{\textrm{high}}=$" + f"{num_arms_w_high_reward}, " + \
        fr"$H \sim {high_reward_rv.to_latex()}$, " + \
        fr"$L \sim {low_reward_rv.to_latex()}$, " + \
        r"$W= {}$, ".format(win_len) + \
        r"$T= {}$, ".format(num_rounds) + \
        r"$N_{\textrm{sim}}= $" + "{}".format(num_sim_runs)

    plot.plot_mean_sim_result(
        mean_sim_result=mean_sim_result,
        title=title,
        plot_suffix="stationary_bandit",
    )
    # log(INFO, "", mean_sim_result=mean_sim_result)


def test_non_stationary_bandit_w_ts_sliding_vs_reset_win(
    num_arms_and_w_high_and_medium_reward: int,
    high_and_medium_and_low_reward_rv: Tuple[rv.RandomVariable, rv.RandomVariable, rv.RandomVariable],
    phase_duration_rv: rv.RandomVariable,
):
    log(INFO, "Started",
        num_arms_and_w_high_and_medium_reward=num_arms_and_w_high_and_medium_reward,
        high_and_medium_and_low_reward_rv=high_and_medium_and_low_reward_rv,
        phase_duration_rv=phase_duration_rv,
    )

    num_arms, num_arms_w_high_reward, num_arms_w_medium_reward = num_arms_and_w_high_and_medium_reward
    high_reward_rv, medium_reward_rv, low_reward_rv = high_and_medium_and_low_reward_rv

    bandit = bandit_module.NonStationaryBandit(
        num_arms=num_arms,
        num_arms_w_high_reward=num_arms_w_high_reward,
        num_arms_w_medium_reward=num_arms_w_medium_reward,
        high_reward_rv=high_reward_rv,
        medium_reward_rv=medium_reward_rv,
        low_reward_rv=low_reward_rv,
        phase_duration_rv=phase_duration_rv,
    )

    agent_ts = agent_module.ThompsonSamplingAgent_full(
        name="TS",
        num_arms=num_arms,
    )

    win_len = 100
    agent_ts_sliding_win = agent_module.ThompsonSamplingAgent_slidingWin(
        name="TS-SlidingWin",
        num_arms=num_arms,
        win_len=win_len,
    )

    agent_ts_reset_win = agent_module.ThompsonSamplingAgent_resetWinOnRareEvent(
        name="TS-ResetWin",
        num_arms=num_arms,
        win_len=win_len,
        tail_mass_threshold=0.05,
    )

    num_rounds = 1000
    num_sim_runs = 2
    mean_sim_result = sim.sim(
        bandit=bandit,
        agent_list=[agent_ts],
        # agent_list=[agent_ts, agent_ts_sliding_win],
        # agent_list=[agent_ts_sliding_win, agent_ts_reset_win],
        num_rounds=num_rounds,
        num_sim_runs=num_sim_runs,
    )

    title = r"$N_{\textrm{arm}}=$" + f"{num_arms}, " + \
        r"$N_{\textrm{high}}=$" + f"{num_arms_w_high_reward}, " + \
        fr"$H \sim {high_reward_rv.to_latex()}$, " + \
        fr"$M \sim {medium_reward_rv.to_latex()}$, " + \
        fr"$L \sim {low_reward_rv.to_latex()}$, " + \
        r"$W= {}$, ".format(win_len) + \
        r"$T= {}$, ".format(num_rounds) + \
        r"$N_{\textrm{sim}}= $" + "{}".format(num_sim_runs)

    plot.plot_mean_sim_result(
        mean_sim_result=mean_sim_result,
        title=title,
        plot_suffix="non_stationary_bandit",
    )
    # log(INFO, "", mean_sim_result=mean_sim_result)
