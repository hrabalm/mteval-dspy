from mteval_dspy.utils import clip


def test_clip_within_bounds():
    assert clip(0.5, 0.0, 1.0) == 0.5


def test_clip_below_bounds():
    assert clip(-1.0, 0.0, 1.0) == 0.0


def test_clip_above_bounds():
    assert clip(2.0, 0.0, 1.0) == 1.0
