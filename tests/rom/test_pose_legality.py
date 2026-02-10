import numpy as np

from smii.rom.pose_legality import LegalityConfig, score_legality


def test_extreme_shoulder_penetrates_more_than_neutral():
    cfg = LegalityConfig(alpha=1.0)
    # Neutral-ish positions: shoulders away from torso.
    joints_neutral = {
        "left_shoulder": np.array([0.2, 0.3, 0.0]),
        "right_shoulder": np.array([-0.2, 0.3, 0.0]),
        "spine1": np.array([0.0, 0.25, 0.0]),
    }
    neutral = score_legality(joints_neutral, cfg)

    # Force shoulders into torso (closer to spine).
    joints_collide = {
        "left_shoulder": np.array([0.05, 0.25, 0.0]),
        "right_shoulder": np.array([-0.05, 0.25, 0.0]),
        "spine1": np.array([0.0, 0.25, 0.0]),
    }
    collide = score_legality(joints_collide, cfg)

    assert collide.score > neutral.score
    assert collide.violations >= neutral.violations
