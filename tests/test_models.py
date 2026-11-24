import mlx.core as mx
import numpy as np

from eval.closed_loop.policies import Observation, PilotNetPolicy
from models.refiner.refiner import Refiner, flow_matching_loss, save_refiner
from models.refiner.unet import RefinerUNet


def test_refiner_shapes_and_roundtrip(tmp_path):
    m = RefinerUNet(base=8)
    x = mx.random.uniform(shape=(2, 36, 52, 3))          # not a multiple of 8: exercises padding
    cond = mx.random.uniform(shape=(2, 36, 52, 7))
    v = m(x, cond, mx.array([0.1, 0.7]))
    assert v.shape == (2, 36, 52, 3)
    assert np.isfinite(flow_matching_loss(m, x, x, cond).item())
    save_refiner(m, tmp_path / "r.safetensors", steps=2)
    r = Refiner.load(tmp_path / "r.safetensors")
    out = r(x[0], mx.ones((36, 52)) * 10, mx.ones((36, 52)))
    assert out.shape == (36, 52, 3)


def test_pilotnet_roundtrip(tmp_path):
    p = PilotNetPolicy()
    obs = Observation(np.random.default_rng(0).uniform(size=(288, 512, 3)).astype(np.float32), 8.0, 0.0)
    a = p.act(obs)
    p.save(tmp_path / "p.safetensors")
    b = PilotNetPolicy.load(tmp_path / "p.safetensors").act(obs)
    assert abs(a.steer - b.steer) < 1e-5 and abs(a.accel - b.accel) < 1e-5
