import abc

from src.prob import rv


class Arm:
    def __init__(self, name: str):
        self.name = name

    @abc.abstractmethod
    def pull(self):
        pass


class StationaryArm(Arm):
    def __init__(self, name: str, reward_rv: rv.RandomVariable):
        super().__init__(name=name)

        self.reward_rv = reward_rv

    def pull(self) -> float:
        return self.reward_rv.sample()


class ArmState(enum.Enum):
    high = "high"
    low = "low"


class NonStationaryArm_wHighLowReward(Arm):
    def __init__(
        self,
        name: str,
        high_reward_rv: rv.RandomVariable,
        low_reward_rv: rv.RandomVariable,
        phase_duration_rv: rv.RandomVariable,
    ):
        super().__init__(name=name)

        self.high_reward_rv = high_reward_rv
        self.low_reward_rv = low_reward_rv
        self.phase_duration_rv = phase_duration_rv

        self.state = ArmState.high
        self.phase_duration = 0

    def __repr__(self):
        return (
            "NonStationaryArm_wHighLowReward( \n"
            f"\t high_reward_rv= {self.high_reward_rv} \n"
            f"\t low_reward_rv= {self.low_reward_rv} \n"
            f"\t phase_duration_rv= {self.phase_duration_rv} \n"
            ")"
        )

    def sample_reward(self) -> float:
        if self.state == ArmState.high:
            return self.high_reward_rv.sample()
        elif self.state == ArmState.low:
            return self.low_reward_rv.sample()

    def switch_state(self):
        if self.state == ArmState.high:
            self.state = ArmState.low
        elif self.state == ArmState.low:
            self.state = ArmState.high

    def pull(self) -> float:
        if self.phase_duration == 0:
            self.phase_duration = int(self.phase_duration_rv.sample())
            self.switch_state()
        else:
            self.phase_duration -= 1

        return self.sample_reward()
