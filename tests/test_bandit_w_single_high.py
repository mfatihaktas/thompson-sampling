import pytest

from typing import Tuple

from src.agent import agent as agent_module
from src.env import bandit as bandit_module
from src.prob import rv
from src.sim import sim
from src.utils.debug import *


@pytest.fixture(
    params=[
        (2, 1),
    ]
)
def num_arms_and_w_high_reward(request) -> Tuple[int, int]:
    return request.param


@pytest.fixture(
    params=[
        (rv.Normal(mu=10, sigma=1), rv.Normal(mu=1, sigma=1)),
    ]
)
def high_and_low_reward_rv(request) -> Tuple[rv.RandomVariable, rv.RandomVariable]:
    return request.param


# @pytest.mark.parametrize(
#     "num_arms",
#     (2, 3),
#     indirect=True
# )
def test_stationary_bandit_w_ts_sliding_win_vs_reset_win(
    num_arms_and_w_high_reward: int,
    high_and_low_reward_rv: Tuple[rv.RandomVariable, rv.RandomVariable],
):
    log(INFO, "Started", num_arms_and_w_high_reward=num_arms_and_w_high_reward, high_and_low_reward_rv=high_and_low_reward_rv)

    num_arms, num_arms_w_high_reward = num_arms_and_w_high_reward
    high_reward_rv, low_reward_rv = high_and_low_reward_rv

    bandit = bandit_module.StationaryBandit(
        num_arms=num_arms,
        num_arms_w_high_reward=num_arms_w_high_reward,
        high_reward_rv=high_reward_rv,
        low_reward_rv=low_reward_rv,
    )

    win_len = 20
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

    sim_result = sim.sim(
        bandit=bandit,
        agent_list=[agent_ts_sliding_win, agent_ts_reset_win],
        num_rounds=10,
    )

    log(INFO, "", sim_result=sim_result)
