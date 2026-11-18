import math

import numpy as np
import pytest
import torch
from torch.nn import functional as F

from src.models.face import FaceModel, IR18, ArcFace
from src.objectives.losses import composition_loss, pair_objective
from src.objectives.pairwise import normalize_embeddings, pairwise_kernel


def test_both_endpoints_have_genuine_and_impostor_gradients():
    torch.manual_seed(20261111)
    z = torch.randn(3, 2, 5, dtype=torch.float64, requires_grad=True)
    h = pairwise_kernel(z, tau=0.1, temperature=0.13)
    single_edge = torch.autograd.grad(h[0, 1], z, retain_graph=True)[0]
    assert torch.all(torch.linalg.vector_norm(single_edge[:2], dim=-1) > 1e-6)
    assert torch.count_nonzero(single_edge[2]) == 0
    # Symmetric aggregation gives the same derivative in either ordered edge.
    torch.testing.assert_close(single_edge, torch.autograd.grad(h[1, 0], z)[0])


def test_encoder_features_end_to_end_envelope_finite_difference():
    gen = torch.Generator().manual_seed(11)
    z = torch.randn(7, 2, 8, generator=gen, dtype=torch.float64, requires_grad=True)
    direction = torch.randn(z.shape, generator=gen, dtype=z.dtype)
    f = lambda x: composition_loss(pairwise_kernel(x, 0.2, 0.07), 0.25)[0]
    loss = f(z)
    (gradient,) = torch.autograd.grad(loss, z)
    assert torch.all(torch.linalg.vector_norm(gradient, dim=-1) > 1e-7)
    step = 1e-6
    with torch.no_grad():
        numeric = (f(z + step * direction) - f(z - step * direction)) / (2 * step)
    torch.testing.assert_close((gradient * direction).sum(), numeric, atol=1e-9, rtol=1e-7)


def test_three_objectives_shape_dtype_and_gradient():
    z = torch.randn(4, 2, 10, generator=torch.Generator().manual_seed(2), requires_grad=True)
    outputs = {}
    for arm in ("nominal", "row", "c2"):
        loss, oracle = pair_objective(z, arm, 0.1, 0.1, 0.05)
        (gradient,) = torch.autograd.grad(loss, z)
        assert loss.ndim == 0 and loss.dtype == torch.float32 and loss.device == z.device
        assert torch.isfinite(loss) and torch.isfinite(gradient).all()
        assert (oracle is not None) == (arm == "c2")
        outputs[arm] = (loss, gradient)
    c2_at_zero, _ = pair_objective(z, "c2", 0.1, 0.0, 0.05)
    torch.testing.assert_close(c2_at_zero, outputs["nominal"][0])
    assert c2_at_zero.item() <= outputs["c2"][0].item()
    torch.testing.assert_close(torch.autograd.grad(c2_at_zero, z)[0], outputs["nominal"][1])


def test_arcface_formula_non_target_and_stability_at_poles():
    head = ArcFace(4, 3).double()
    with torch.no_grad():
        head.weight.copy_(torch.tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1], [-1, 0, 0]]))
    features = torch.tensor([[0.4, 1, 2], [1, 0, 0], [1, 0, 0], [0, 0, 0]],
                            dtype=torch.float64, requires_grad=True)
    labels = torch.tensor([0, 0, 3, 2])
    logits = head(features, labels)
    cosine = normalize_embeddings(features) @ normalize_embeddings(head.weight).T
    assert logits.shape == (4, 4)
    indices = F.one_hot(labels, num_classes=4).bool()
    torch.testing.assert_close(logits[~indices], (64 * cosine)[~indices])
    assert logits[0, 0].item() == pytest.approx(64 * math.cos(math.acos(cosine[0, 0].item()) + 0.5))
    assert logits[2, 3].item() == pytest.approx(64 * (-1 - math.sin(0.5) * 0.5))
    F.cross_entropy(logits, labels).backward()
    assert torch.isfinite(features.grad).all()
    assert torch.isfinite(head.weight.grad).all()


def test_normalizer_zero_huge_and_malformed_inputs():
    x = torch.tensor([[0, 0, 0], [1e30, -1e30, 0], [1e-22, 0, 1e-22]], requires_grad=True)
    z = normalize_embeddings(x)
    torch.testing.assert_close(torch.linalg.vector_norm(z, dim=1), torch.ones(3))
    torch.testing.assert_close(z[0], torch.tensor([1.0, 0.0, 0.0]))
    z.sum().backward()
    assert torch.isfinite(x.grad).all()
    with pytest.raises(ValueError, match="two distinct"):
        pairwise_kernel(torch.ones(4, 3, 7), 0.1)


@pytest.mark.parametrize("bad", [float("nan"), float("inf")])
def test_nonfinite_pair_features_are_not_accepted_by_solver(bad):
    x = torch.randn(4, 2, 6)
    x[0, 0, 0] = bad
    with pytest.raises(ValueError, match="finite"):
        pair_objective(x, "c2", 0.1, 0.1, 0.05)


def test_full_ir18_gradients_batchnorm_and_inference():
    torch.manual_seed(3)
    model = FaceModel(6).train()
    images = torch.randn(6, 3, 112, 112)
    bn = model.encoder.stem[1]
    previous = bn.running_mean.clone()
    features = model.encoder(images)
    assert features.shape == (6, 512)
    assert not torch.equal(previous, bn.running_mean)
    pair, _ = pair_objective(features.reshape(3, 2, 512), "c2", 0.1, 0.1, 0.05)
    pair.backward()
    stem_grad = model.encoder.stem[0].weight.grad
    assert stem_grad is not None
    assert torch.isfinite(stem_grad).all() and stem_grad.norm() > 0
    assert model.head.weight.grad is None
    model.eval()
    running = bn.running_mean.clone()
    with torch.no_grad():
        z = model.encoder.embed(images)
        torch.testing.assert_close(z, model.encoder.embed(images))
    torch.testing.assert_close(z.norm(dim=1), torch.ones(6))
    torch.testing.assert_close(bn.running_mean, running)
    with pytest.raises(ValueError, match="112"):
        IR18()(torch.zeros(2, 3, 32, 32))


def test_encoder_cpu_amp_uses_float32_pair_and_head():
    torch.manual_seed(4)
    model = FaceModel(4)
    labels = torch.tensor([0, 0, 1, 1, 2, 2])
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        features = model.encoder(torch.randn(6, 3, 112, 112))
        logits = model.head(features, labels)
        h = pairwise_kernel(features.reshape(3, 2, 512), 0.1)
    assert features.dtype == torch.bfloat16
    assert logits.dtype == h.dtype == torch.float32
    risk, oracle = composition_loss(h, 0.1)
    assert oracle.q_signed.dtype == np.float64
    (F.cross_entropy(logits, labels) + risk).backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a CUDA GPU")
@pytest.mark.parametrize("dtype", [
    pytest.param(torch.float16, id="fp16"),
    pytest.param(torch.bfloat16, id="bf16"),
])
def test_cuda_amp_global_oracle_and_optimizer_step(dtype):
    if dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
        pytest.skip("CUDA device lacks bfloat16 support")
    torch.manual_seed(41)
    model = FaceModel(4).cuda()
    inputs = torch.randn(8, 3, 112, 112, device="cuda")
    labels = torch.arange(4, device="cuda").repeat_interleave(2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    scaler = torch.amp.GradScaler("cuda", enabled=dtype == torch.float16, init_scale=128)
    old_weight = model.encoder.stem[0].weight.detach().clone()
    with torch.autocast("cuda", dtype=dtype):
        features = model.encoder(inputs)
    pair, oracle = pair_objective(features.reshape(4, 2, 512), "c2", 0.1, 0.1, 0.05)
    logits = model.head(features, labels)
    assert features.dtype == dtype
    assert logits.dtype == pair.dtype == torch.float32
    assert pair.is_cuda and oracle.q_signed.dtype == np.float64 and oracle.nonnegative_witness
    reference, _ = pair_objective(features.float().reshape(4, 2, 512), "c2", 0.1, 0.1, 0.05)
    torch.testing.assert_close(pair, reference)
    scaler.scale(F.cross_entropy(logits, labels) + pair).backward()
    scaler.unscale_(optimizer)
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    scaler.step(optimizer)
    scaler.update()
    assert not torch.equal(old_weight, model.encoder.stem[0].weight)
