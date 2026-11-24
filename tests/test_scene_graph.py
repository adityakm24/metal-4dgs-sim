import numpy as np

from capture.synthetic import build_world
from engine.scene_graph import SceneGraph, Track, render, yaw_quat
from eval.closed_loop import perturbations as P


def small_world():
    return build_world(length=40, duration=3.0, density=0.15, width=128, height=72)


def test_track_interpolation_and_edits():
    tr = Track([0, 1, 2], [[0, 0, 0], [10, 0, 0], [20, 0, 0]], [yaw_quat(0)] * 3)
    assert np.allclose(tr.position(0.5), [5, 0, 0])
    assert np.allclose(tr.position(-1), [0, 0, 0]) and np.allclose(tr.position(5), [20, 0, 0])
    lo = tr.lateral_offset(-2.0, t_start=0.5, ramp=0.5)
    assert abs(lo.position(1.5)[1] + 2.0) < 1e-6 and abs(lo.position(0.2)[1]) < 1e-9
    assert np.allclose(tr.time_shift(1.0).position(1.5), [5, 0, 0])
    hb = tr.hard_brake(1.0, decel=10.0)
    assert hb.position(2.0)[0] < 20.0


def test_save_load_roundtrip_and_edits(tmp_path):
    w = small_world()
    p = w.save(tmp_path / "s.safetensors")
    s2 = SceneGraph.load(p)
    assert s2.num_gaussians == w.num_gaussians
    assert np.allclose(s2.node("actor_1").track.pos, w.node("actor_1").track.pos)
    g = s2.graph_json()
    g["nodes"][1]["track"] = Track.static([1, 2, 3]).to_json()
    import json
    SceneGraph.edits_path(p).write_text(json.dumps(g))
    s3 = SceneGraph.load(p)
    assert np.allclose(s3.node("actor_1").track.position(1.0), [1, 2, 3])
    assert np.allclose(SceneGraph.load(p, edits=False).node("actor_1").track.pos, w.node("actor_1").track.pos)


def test_render_outputs_and_perturbation():
    w = small_world()
    cam = w.ego_camera(1.0)
    out = render(w, cam, 1.0, flow_ref=(w.ego_camera(0.9), 0.9))
    d = out.numpy()
    assert d["rgb"].shape == (72, 128, 3) and d["flow"].shape == (72, 128, 2)
    assert d["alpha"].mean() > 0.3 and d["depth"].max() > 1.0
    specs = P.parse("actor_2:lateral_offset=-1.8m,t=1.0s;ego:yaw=5deg;camera:occlusion=0.3,t=1s;obstacle_1:spawn=20m,t=0.5s")
    assert specs[0].value == -1.8 and abs(specs[1].value - np.deg2rad(5)) < 1e-12
    sc, ego, fx = P.apply(w, specs)
    assert abs(sc.node("actor_2").track.position(2.5)[1] - (w.node("actor_2").track.position(2.5)[1] - 1.8)) < 1e-6
    assert np.allclose(w.node("actor_2").track.position(2.5)[1], 3.5)   # base scene untouched
    assert "obstacle_1" in [n.name for n in sc.nodes] and len(fx) == 1
