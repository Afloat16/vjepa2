# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from copy import deepcopy

import pytest
import torch

from src.models.utils.modules import ACRoPEAttention, Attention, RoPEAttention


CLASSES = [Attention, RoPEAttention, ACRoPEAttention]


def forward(module, x):
    if isinstance(module, ACRoPEAttention):
        return module(x, T=2, H=2, W=2)
    if isinstance(module, RoPEAttention):
        return module(x, T=2, H_patches=2, W_patches=2)
    return module(x)


@pytest.mark.parametrize("cls", CLASSES)
def test_custom_scale_matches_explicit_attention_values_and_gradients(cls):
    torch.manual_seed(42)
    native = cls(dim=72, num_heads=4, qkv_bias=True, qk_scale=0.7).double().eval()
    explicit = deepcopy(native)
    explicit.use_sdpa = False
    x = torch.randn(2, 8, 72, dtype=torch.float64, requires_grad=True)
    reference_x = x.detach().clone().requires_grad_()
    actual = forward(native, x)
    expected = forward(explicit, reference_x)
    torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)
    actual_grad = torch.autograd.grad(actual.square().sum(), (x, native.qkv.weight))
    expected_grad = torch.autograd.grad(expected.square().sum(), (reference_x, explicit.qkv.weight))
    for result, reference in zip(actual_grad, expected_grad):
        torch.testing.assert_close(result, reference, atol=1e-11, rtol=1e-11)


@pytest.mark.parametrize("cls", CLASSES)
def test_eval_disables_attention_and_projection_dropout(cls):
    torch.manual_seed(42)
    native = cls(dim=72, num_heads=4, attn_drop=0.3, proj_drop=0.7).double().eval()
    explicit = deepcopy(native)
    explicit.use_sdpa = False
    x = torch.randn(2, 8, 72, dtype=torch.float64)
    first = forward(native, x)
    second = forward(native, x)
    expected = forward(explicit, x)
    torch.testing.assert_close(first, second, atol=0, rtol=0)
    torch.testing.assert_close(first, expected, atol=1e-12, rtol=1e-12)


@pytest.mark.parametrize("cls", CLASSES)
def test_training_uses_attention_dropout_probability(cls):
    torch.manual_seed(42)
    native = cls(dim=72, num_heads=4, attn_drop=0.4, proj_drop=0.0).double().train()
    explicit = deepcopy(native)
    explicit.use_sdpa = False
    x = torch.randn(2, 8, 72, dtype=torch.float64)
    # CPU math SDPA and explicit attention apply dropout to the same [B,H,N,N] probabilities.
    with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
        torch.manual_seed(7)
        actual = forward(native, x)
        torch.manual_seed(7)
        expected = forward(explicit, x)
    torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)


@pytest.mark.parametrize("cls", CLASSES)
def test_default_zero_dropout_backend_behavior_is_preserved(cls):
    torch.manual_seed(42)
    native = cls(dim=72, num_heads=4).double().eval()
    explicit = deepcopy(native)
    explicit.use_sdpa = False
    x = torch.randn(2, 8, 72, dtype=torch.float64)
    torch.testing.assert_close(forward(native, x), forward(explicit, x), atol=1e-12, rtol=1e-12)
