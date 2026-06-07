import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("Box2D")
gym = pytest.importorskip("gymnasium")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def test_falcon9_is_gymnasium_env_with_gymnasium_spaces():
    from gymnasium import spaces
    from falcon9 import Falcon9

    env = Falcon9()

    assert isinstance(env, gym.Env)
    assert isinstance(env.observation_space, spaces.Box)
    assert isinstance(env.action_space, spaces.Discrete)
    assert env.observation_space.shape == (11,)
    assert env.action_space.n == 4
    env.close()


def test_falcon9_reset_and_step_follow_gymnasium_api():
    from falcon9 import Falcon9

    env = Falcon9(dashboard=True)
    obs, info = env.reset(seed=123)

    assert env.observation_space.contains(obs)
    assert isinstance(info, dict)

    next_obs, reward, terminated, truncated, step_info = env.step(env.action_space.sample())

    assert env.observation_space.contains(next_obs)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert "success" in step_info
    env.close()


def test_falcon9_continuous_action_space_uses_two_controls():
    from gymnasium import spaces
    from falcon9 import Falcon9

    env = Falcon9(continuous=True)

    assert isinstance(env.action_space, spaces.Box)
    assert env.action_space.shape == (2,)
    assert env.action_space.contains(np.array([0.0, 0.0], dtype=np.float32))
    env.close()


def test_falcon9_registers_with_gym_make():
    import falcon9  # noqa: F401

    env = gym.make("Falcon9-v0", dashboard=True)
    obs, info = env.reset(seed=5)

    assert env.observation_space.contains(obs)
    assert isinstance(info, dict)
    assert env.spec.max_episode_steps is None
    assert env.unwrapped.max_episode_steps == 1200
    env.close()


def test_hard_platform_impact_is_failure():
    from falcon9 import Falcon9

    env = Falcon9()
    state, _ = env.reset(seed=7)
    env.left_foot_contact = True
    env.right_foot_contact = True
    env.platform_contact = True
    env.anchored_feet = {"left_foot", "right_foot"}
    env.platform_impact_velocity = (0.0, -12.0, 0.0)

    terminated, success, failure_reason = env._terminal_status(state)

    assert terminated is True
    assert success is False
    assert failure_reason == "hard_platform_impact"
    env.close()


def test_two_anchored_feet_do_not_immediately_succeed():
    from falcon9 import Falcon9

    env = Falcon9()
    state, _ = env.reset(seed=9)
    env.left_foot_contact = True
    env.right_foot_contact = True
    env.platform_contact = True
    env.anchored_feet = {"left_foot", "right_foot"}
    env.platform_impact_velocity = (0.0, -0.5, 0.0)
    env.booster.awake = True

    terminated, success, failure_reason = env._terminal_status(state)

    assert terminated is False
    assert success is False
    assert failure_reason is None
    env.close()
