"""unit tests for the FTG evaluation smoothness metrics."""

import numpy as np
import pytest
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from evaluate_baseline import run_episode, summarize


class FakeDriver:
    """Return predefined actions instead of using the FTG algorithm."""

    def __init__(self, actions):
        self.actions = actions
        self.index = 0

    def reset(self):
        self.index = 0

    def act(self, obs):
        action = self.actions[self.index]
        self.index += 1
        return action


class FakeEnv:
    """A small fake environment that runs a fixed number of steps."""

    def __init__(self, num_steps):
        self.num_steps = num_steps
        self.steps = 0
        self.received_actions = []

    def reset(self, seed=None, options=None):
        self.steps = 0
        self.received_actions = []
        return {}, {'progress': 0.0}

    def step(self, action):
        self.received_actions.append(action)
        self.steps += 1

        done = self.steps >= self.num_steps

        info = {
            'wall_collision': False,
            'progress': self.steps * 0.1,
            'lap': 1,
            'time': self.steps * 0.01,
        }

        return {}, 0.0, done, False, info


def make_action(motor, steering):
    return {
        'motor': np.array([motor], dtype=float),
        'steering': np.array([steering], dtype=float),
    }


def test_smoothness_calculation():
    actions = [
        make_action(0.0, 0.1),
        make_action(0.4, -0.1),
        make_action(0.2, 0.2),
    ]

    env = FakeEnv(num_steps=3)
    driver = FakeDriver(actions)

    result = run_episode(env, driver, seed=100)

    # Motor differences: 0.4 and 0.2 -> average 0.3
    assert result['motor_smoothness'] == pytest.approx(0.3)

    # Steering differences: 0.2 and 0.3 -> average 0.25
    assert result['steering_smoothness'] == pytest.approx(0.25)

    # The evaluator must pass the original actions to the environment.
    assert all(
        received is original
        for received, original in zip(env.received_actions, actions)
    )
    assert len(env.received_actions) == 3


def test_constant_actions_have_zero_smoothness():
    actions = [
        make_action(0.5, 0.2),
        make_action(0.5, 0.2),
        make_action(0.5, 0.2),
    ]

    result = run_episode(FakeEnv(3), FakeDriver(actions), seed=101)

    assert result['motor_smoothness'] == 0.0
    assert result['steering_smoothness'] == 0.0


def test_single_action_has_undefined_smoothness():
    actions = [make_action(0.5, 0.1)]

    result = run_episode(FakeEnv(1), FakeDriver(actions), seed=102)

    assert result['motor_smoothness'] is None
    assert result['steering_smoothness'] is None


def test_summary_includes_smoothness():
    episodes = [
        {
            'seed': 100,
            'lap_completed': True,
            'lap_time': 80.0,
            'collisions': 0,
            'motor_smoothness': 0.2,
            'steering_smoothness': 0.4,
        },
        {
            'seed': 101,
            'lap_completed': True,
            'lap_time': 82.0,
            'collisions': 0,
            'motor_smoothness': 0.4,
            'steering_smoothness': 0.2,
        },
    ]

    result = summarize(episodes)

    assert result['motor_smoothness_mean'] == pytest.approx(0.3)
    assert result['steering_smoothness_mean'] == pytest.approx(0.3)
    assert result['laps_completed'] == '2/2'