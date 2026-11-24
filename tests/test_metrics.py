import numpy as np

from eval.closed_loop.dynamics import Action, BicycleModel, EgoState
from eval.metrics.geometry import warping_error
from eval.metrics.safety import ReferencePath, obb_overlap, time_to_collision
from engine.scene_graph.tracks import pose_matrix, yaw_quat


def test_obb_and_ttc():
    a = pose_matrix([0, 0, 0], yaw_quat(0))
    b = pose_matrix([4.0, 0.5, 0], yaw_quat(0.3))
    assert obb_overlap(a, [4.5, 1.9], b, [4.5, 1.9])
    c = pose_matrix([20.0, 0, 0], yaw_quat(0))
    assert not obb_overlap(a, [4.5, 1.9], c, [4.5, 1.9])
    ttc = time_to_collision(a, [4.5, 1.9], 10.0, c, [4.5, 1.9], np.array([5.0, 0, 0]))
    assert abs(ttc - (20 - 4.5) / 5.0) < 1e-6


def test_bicycle_straight_and_turn():
    m = BicycleModel()
    s = EgoState(0, 0, 0, 10)
    for _ in range(10):
        s = m.step(s, Action(0, 0), 0.1)
    assert abs(s.x - 10) < 1e-6 and abs(s.y) < 1e-9
    s2 = m.step(EgoState(0, 0, 0, 10), Action(0.2, 0), 0.1)
    assert s2.yaw > 0 and s2.y > 0


def test_reference_path():
    rp = ReferencePath(np.array([[0, 0], [10, 0], [20, 0]]))
    s, lat, _ = rp.project([5, 1.5])
    assert abs(s - 5) < 1e-9 and abs(lat - 1.5) < 1e-9


def test_warping_error_zero_for_pure_shift():
    rng = np.random.default_rng(0)
    prev = rng.uniform(size=(20, 30, 3)).astype(np.float32)
    curr = np.zeros_like(prev)
    curr[:, :-2] = prev[:, 2:]          # content moved 2 px left
    flow = np.zeros((20, 30, 2), np.float32)
    flow[..., 0] = 2.0                  # pixel x in curr was at x+2 in prev
    assert warping_error(curr, prev, flow) < 1e-6
