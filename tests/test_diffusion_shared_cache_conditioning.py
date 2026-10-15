# Copyright 2024 ByteDance and/or its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from copy import deepcopy
from types import SimpleNamespace
import unittest
from unittest import mock

import torch
from torch.testing import assert_close

from protenix.model.modules.diffusion import DiffusionConditioning
from protenix.model.protenix import Protenix


def _run_conditioning(
    module: DiffusionConditioning,
    inputs: dict[str, torch.Tensor],
    *,
    use_cache: bool,
    use_conditioning: bool,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor | None]]:
    module.train()
    local_inputs = {
        name: tensor.detach().clone().requires_grad_(name != "t_hat_noise_level")
        for name, tensor in inputs.items()
    }
    if use_cache:
        pair_cache = module.prepare_cache(
            relp_feature=local_inputs["relp_feature"],
            z_trunk=local_inputs["z_trunk"],
            use_conditioning=use_conditioning,
        )
        z_argument = None
    else:
        pair_cache = None
        z_argument = local_inputs["z_trunk"]

    single_s, pair_z = module(
        t_hat_noise_level=local_inputs["t_hat_noise_level"],
        relp_feature=local_inputs["relp_feature"],
        s_inputs=local_inputs["s_inputs"],
        s_trunk=local_inputs["s_trunk"],
        z_trunk=z_argument,
        pair_z=pair_cache,
        use_conditioning=use_conditioning,
    )
    single_probe = torch.linspace(
        0.25,
        1.25,
        single_s.numel(),
        device=single_s.device,
        dtype=single_s.dtype,
    ).reshape_as(single_s)
    pair_probe = torch.linspace(
        -0.75,
        0.5,
        pair_z.numel(),
        device=pair_z.device,
        dtype=pair_z.dtype,
    ).reshape_as(pair_z)
    ((single_s * single_probe).sum() + (pair_z * pair_probe).sum()).backward()

    gradients: dict[str, torch.Tensor | None] = {
        name: (
            None
            if tensor.grad is None
            else tensor.grad.detach().clone()
        )
        for name, tensor in local_inputs.items()
        if name != "t_hat_noise_level"
    }
    gradients.update(
        {
            f"parameter:{name}": (
                None if parameter.grad is None else parameter.grad.detach().clone()
            )
            for name, parameter in module.named_parameters()
        }
    )
    return single_s.detach(), pair_z.detach(), gradients


@unittest.skipUnless(
    torch.cuda.is_available(),
    "DiffusionConditioning uses the CUDA fused LayerNorm",
)
class TestDiffusionConditioningCacheEquivalence(unittest.TestCase):
    def test_cached_matches_uncached_with_conditioning_on_and_off(self):
        for use_conditioning in (True, False):
            with self.subTest(use_conditioning=use_conditioning):
                device = torch.device("cuda")
                torch.manual_seed(20260724)
                uncached = DiffusionConditioning(
                    c_z=6,
                    c_s=8,
                    c_s_inputs=10,
                    c_noise_embedding=12,
                ).to(device)
                cached = deepcopy(uncached)

                n_sample, n_token = 3, 4
                relp_dim = uncached.relpe.linear_no_bias.weight.shape[1]
                inputs = {
                    "t_hat_noise_level": torch.rand(n_sample, device=device) + 0.5,
                    "relp_feature": torch.randn(
                        n_token, n_token, relp_dim, device=device
                    ),
                    "s_inputs": torch.randn(n_token, 10, device=device),
                    "s_trunk": torch.randn(n_token, 8, device=device),
                    "z_trunk": torch.randn(n_token, n_token, 6, device=device),
                }

                uncached_s, uncached_z, uncached_grads = _run_conditioning(
                    uncached,
                    inputs,
                    use_cache=False,
                    use_conditioning=use_conditioning,
                )
                cached_s, cached_z, cached_grads = _run_conditioning(
                    cached,
                    inputs,
                    use_cache=True,
                    use_conditioning=use_conditioning,
                )

                assert_close(cached_s, uncached_s, rtol=1e-6, atol=1e-6)
                assert_close(cached_z, uncached_z, rtol=1e-6, atol=1e-6)
                self.assertEqual(cached_grads.keys(), uncached_grads.keys())
                for name in cached_grads:
                    cached_grad = cached_grads[name]
                    uncached_grad = uncached_grads[name]
                    self.assertEqual(
                        cached_grad is None,
                        uncached_grad is None,
                        name,
                    )
                    if cached_grad is not None:
                        assert_close(
                            cached_grad,
                            uncached_grad,
                            rtol=1e-5,
                            atol=1e-6,
                        )

                if not use_conditioning:
                    assert_close(
                        cached_grads["s_trunk"],
                        torch.zeros_like(cached_grads["s_trunk"]),
                    )
                    assert_close(
                        cached_grads["z_trunk"],
                        torch.zeros_like(cached_grads["z_trunk"]),
                    )


class _DummyDiffusionConditioning(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(1.25))
        self.calls: list[bool] = []

    def prepare_cache(
        self,
        relp_feature,
        z_trunk,
        inplace_safe=False,
        use_conditioning=True,
    ):
        del inplace_safe
        self.calls.append(use_conditioning)
        if not use_conditioning:
            z_trunk = 0 * z_trunk
        return self.scale * z_trunk + relp_feature


class _DummyAtomEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(0.75))
        self.calls = 0

    def prepare_cache(self, *, z, ref_pos, **kwargs):
        del kwargs
        self.calls += 1
        return self.scale * z, ref_pos


class _DummyDiffusionModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.diffusion_conditioning = _DummyDiffusionConditioning()
        self.atom_attention_encoder = _DummyAtomEncoder()
        self.training_chunk_cache_ids: list[tuple[int, int, int]] = []
        self.training_chunk_cache_ptrs: list[tuple[int, int, int]] = []


class _DummyPermutation:
    def permute_label_to_match_mini_rollout(
        self,
        coordinate_mini,
        input_feature_dict,
        label_dict,
        label_full_dict,
    ):
        del coordinate_mini, input_feature_dict, label_full_dict
        return label_dict, {}

    def permute_diffusion_sample_to_match_label(
        self,
        input_feature_dict,
        pred_dict,
        label_dict,
        stage,
    ):
        del input_feature_dict, label_dict, stage
        return pred_dict, {}, None, None


class _TrainLoopHarness:
    def __init__(self, *, enable_cache: bool):
        self.enable_diffusion_shared_vars_cache = enable_cache
        self.enable_efficient_fusion = False
        self.train_confidence_only = False
        self.diffusion_batch_size = 6
        self.train_noise_sampler = object()
        self.diffusion_module = _DummyDiffusionModule()
        self.configs = SimpleNamespace(
            skip_amp=SimpleNamespace(
                sample_diffusion=False,
                sample_diffusion_training=False,
            ),
            sample_diffusion={
                "N_sample_mini_rollout": 1,
                "N_step_mini_rollout": 1,
            },
            model=SimpleNamespace(
                confidence_embedding_drop_rate=0.5,
                condition_embedding_drop_rate=0.5,
            ),
            triangle_multiplicative="torch",
            triangle_attention="torch",
            diffusion_chunk_size=2,
        )
        self.s_inputs = torch.randn(3, 2, requires_grad=True)
        self.s_trunk = torch.randn(3, 2, requires_grad=True)
        self.z_trunk = torch.randn(3, 3, 2, requires_grad=True)
        self.mini_cache = None

    def get_pairformer_output(self, **kwargs):
        del kwargs
        return self.s_inputs, self.s_trunk, self.z_trunk

    def _prepare_diffusion_shared_cache(
        self,
        input_feature_dict,
        z_trunk,
        use_conditioning=True,
    ):
        return Protenix._prepare_diffusion_shared_cache(
            self,
            input_feature_dict,
            z_trunk,
            use_conditioning,
        )

    def sample_diffusion(self, **kwargs):
        self.mini_cache = {
            name: kwargs[name] for name in ("pair_z", "p_lm", "c_l")
        }
        n_atom = kwargs["input_feature_dict"]["ref_pos"].shape[-2]
        return torch.zeros(1, n_atom, 3)

    def inference_noise_scheduler(self, N_step, device, dtype):
        del N_step
        return torch.ones(2, device=device, dtype=dtype)

    def run_confidence_head(self, **kwargs):
        del kwargs
        return (torch.tensor(0.0),) * 4

    def distogram_head(self, z):
        return z.sum(dim=-1)


def _fake_sample_diffusion_training(
    *,
    denoise_net,
    input_feature_dict,
    s_inputs,
    s_trunk,
    z_trunk,
    pair_z,
    p_lm,
    c_l,
    N_sample,
    diffusion_chunk_size,
    use_conditioning,
    **kwargs,
):
    del kwargs
    chunks = []
    for start in range(0, N_sample, diffusion_chunk_size):
        chunk_size = min(diffusion_chunk_size, N_sample - start)
        if pair_z is None:
            chunk_pair_z = denoise_net.diffusion_conditioning.prepare_cache(
                input_feature_dict["relp"],
                z_trunk,
                use_conditioning=use_conditioning,
            )
            chunk_p_lm, chunk_c_l = denoise_net.atom_attention_encoder.prepare_cache(
                z=chunk_pair_z,
                ref_pos=input_feature_dict["ref_pos"],
            )
        else:
            chunk_pair_z, chunk_p_lm, chunk_c_l = pair_z, p_lm, c_l
        denoise_net.training_chunk_cache_ids.append(
            (id(chunk_pair_z), id(chunk_p_lm), id(chunk_c_l))
        )
        denoise_net.training_chunk_cache_ptrs.append(
            (
                chunk_pair_z.data_ptr(),
                chunk_p_lm.data_ptr(),
                chunk_c_l.data_ptr(),
            )
        )
        conditioned_s = s_trunk if use_conditioning else 0 * s_trunk
        score = (
            s_inputs.sum()
            + conditioned_s.sum()
            + chunk_pair_z.sum()
            + chunk_p_lm.sum()
            + chunk_c_l.sum()
        )
        n_atom = input_feature_dict["ref_pos"].shape[-2]
        chunks.append(score.expand(chunk_size, n_atom, 3))
    x_denoised = torch.cat(chunks, dim=0)
    return (
        torch.zeros_like(x_denoised),
        x_denoised,
        torch.ones(N_sample),
    )


def _run_train_loop(*, enable_cache: bool, drop_conditioning: bool):
    torch.manual_seed(17)
    harness = _TrainLoopHarness(enable_cache=enable_cache)
    input_feature_dict = {
        "relp": torch.randn_like(harness.z_trunk),
        "ref_pos": torch.randn(4, 3),
        "ref_charge": torch.zeros(4),
        "ref_mask": torch.ones(4),
        "ref_element": torch.zeros(4, 128),
        "ref_atom_name_chars": torch.zeros(4, 4, 64),
        "atom_to_token_idx": torch.tensor([0, 1, 2, 2]),
        "d_lm": torch.zeros(1),
        "v_lm": torch.zeros(1),
        "pad_info": {},
    }
    label_dict = {
        "coordinate": torch.zeros(4, 3),
        "coordinate_mask": torch.ones(4),
    }
    conditioning_draw = 0.1 if drop_conditioning else 0.9
    with (
        mock.patch(
            "protenix.model.protenix.random.random",
            side_effect=[0.9, conditioning_draw],
        ) as random_mock,
        mock.patch(
            "protenix.model.protenix.sample_diffusion_training",
            side_effect=_fake_sample_diffusion_training,
        ),
    ):
        pred_dict, _, _ = Protenix.main_train_loop(
            harness,
            input_feature_dict=input_feature_dict,
            label_full_dict=label_dict,
            label_dict=label_dict,
            N_cycle=1,
            symmetric_permutation=_DummyPermutation(),
        )
    pred_dict["coordinate"].sum().backward()
    gradients = {
        "s_inputs": harness.s_inputs.grad.detach().clone(),
        "s_trunk": harness.s_trunk.grad.detach().clone(),
        "z_trunk": harness.z_trunk.grad.detach().clone(),
        "pair_scale": (
            harness.diffusion_module.diffusion_conditioning.scale.grad.detach().clone()
        ),
        "atom_scale": (
            harness.diffusion_module.atom_attention_encoder.scale.grad.detach().clone()
        ),
    }
    return harness, pred_dict["coordinate"].detach(), gradients, random_mock.call_count


class TestTrainDiffusionSharedCacheLifecycle(unittest.TestCase):
    def test_cache_matches_uncached_and_prepares_once_for_training_chunks(self):
        for drop_conditioning in (False, True):
            with self.subTest(drop_conditioning=drop_conditioning):
                (
                    uncached,
                    uncached_coordinate,
                    uncached_grads,
                    uncached_random_calls,
                ) = _run_train_loop(
                    enable_cache=False,
                    drop_conditioning=drop_conditioning,
                )
                (
                    cached,
                    cached_coordinate,
                    cached_grads,
                    cached_random_calls,
                ) = _run_train_loop(
                    enable_cache=True,
                    drop_conditioning=drop_conditioning,
                )

                assert_close(cached_coordinate, uncached_coordinate)
                for name in cached_grads:
                    assert_close(cached_grads[name], uncached_grads[name])
                self.assertEqual(uncached_random_calls, 2)
                self.assertEqual(cached_random_calls, 2)

                conditioning_calls = (
                    cached.diffusion_module.diffusion_conditioning.calls
                )
                expected_calls = [True, False] if drop_conditioning else [True]
                self.assertEqual(conditioning_calls, expected_calls)
                self.assertEqual(
                    cached.diffusion_module.atom_attention_encoder.calls,
                    len(expected_calls),
                )

                # Mini-rollout always receives the first full-conditioning cache.
                self.assertIsNotNone(cached.mini_cache["pair_z"])
                training_cache_ids = (
                    cached.diffusion_module.training_chunk_cache_ids
                )
                self.assertEqual(len(training_cache_ids), 3)
                self.assertEqual(len(set(training_cache_ids)), 1)
                training_pair_ptr = (
                    cached.diffusion_module.training_chunk_cache_ptrs[0][0]
                )
                if drop_conditioning:
                    self.assertNotEqual(
                        cached.mini_cache["pair_z"].data_ptr(),
                        training_pair_ptr,
                    )
                else:
                    self.assertEqual(
                        cached.mini_cache["pair_z"].data_ptr(),
                        training_pair_ptr,
                    )

                # The uncached reference prepares independently in each chunk.
                self.assertEqual(
                    uncached.diffusion_module.diffusion_conditioning.calls,
                    [not drop_conditioning] * 3,
                )
                self.assertEqual(
                    uncached.diffusion_module.atom_attention_encoder.calls,
                    3,
                )
