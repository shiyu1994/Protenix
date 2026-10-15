from types import SimpleNamespace

import pytest
import torch

from protenix.config.config import parse_configs
from protenix.utils import training as training_utils
from runner.train import (
    build_ddp_kwargs,
    build_run_manifest,
    validate_static_graph_template_usage,
)


def _base_required_args(extra: str = "") -> str:
    return (
        "--project test --run_name fast_options --base_dir /tmp/fast_options "
        "--max_steps 1 --eval_interval 0 --log_interval 1 "
        f"{extra}"
    )


def _ordinary_adam_config(*, fused: bool) -> SimpleNamespace:
    return SimpleNamespace(
        optimizer_sharding="none",
        lr=1e-3,
        adam=SimpleNamespace(
            use_adamw=False,
            fused=fused,
            lr=1e-3,
            weight_decay=0.0,
            beta1=0.9,
            beta2=0.99,
        ),
    )


def test_fast_options_default_to_false():
    from configs.configs_base import configs

    resolved = parse_configs(configs, arg_str=_base_required_args())

    assert resolved.ddp.gradient_as_bucket_view is False
    assert resolved.adam.fused is False


def test_fast_options_parse_explicit_true():
    from configs.configs_base import configs

    resolved = parse_configs(
        configs,
        arg_str=_base_required_args(
            "--ddp.gradient_as_bucket_view true --adam.fused true"
        ),
    )

    assert resolved.ddp.gradient_as_bucket_view is True
    assert resolved.adam.fused is True


def test_ddp_kwargs_forward_gradient_bucket_view():
    configs = SimpleNamespace(
        find_unused_parameters=False,
        ddp=SimpleNamespace(gradient_as_bucket_view=True),
    )

    kwargs = build_ddp_kwargs(configs, local_rank=3)

    assert kwargs == {
        "find_unused_parameters": False,
        "device_ids": [3],
        "output_device": 3,
        "static_graph": True,
        "gradient_as_bucket_view": True,
    }


def test_ddp_kwargs_preserve_false_default_when_section_is_missing():
    kwargs = build_ddp_kwargs(
        SimpleNamespace(find_unused_parameters=False),
        local_rank=0,
    )

    assert kwargs["gradient_as_bucket_view"] is False


def _mixed_template_configs(*, emit_dummy_on_disabled):
    return SimpleNamespace(
        find_unused_parameters=False,
        ddp=SimpleNamespace(gradient_as_bucket_view=True),
        data=SimpleNamespace(
            train_sets=["pdb", "distillation"],
            template=SimpleNamespace(enable_prot_template=True),
            pdb=SimpleNamespace(template=SimpleNamespace()),
            distillation=SimpleNamespace(
                template=SimpleNamespace(
                    enable_prot_template=False,
                    emit_dummy_template_when_disabled=(
                        emit_dummy_on_disabled
                    ),
                )
            ),
        ),
    )


def test_static_graph_rejects_mixed_template_parameter_usage():
    configs = _mixed_template_configs(emit_dummy_on_disabled=False)
    with pytest.raises(
        RuntimeError,
        match="invariant TemplateEmbedder parameter usage",
    ):
        build_ddp_kwargs(configs, local_rank=0)


def test_static_graph_accepts_disabled_source_with_dummy_template():
    configs = _mixed_template_configs(emit_dummy_on_disabled=True)
    assert validate_static_graph_template_usage(configs) == {
        "pdb": True,
        "distillation": True,
    }
    kwargs = build_ddp_kwargs(configs, local_rank=0)
    assert kwargs["static_graph"] is True
    assert kwargs["find_unused_parameters"] is False


def test_static_graph_accepts_all_sources_skipping_template_embedder():
    configs = _mixed_template_configs(emit_dummy_on_disabled=False)
    configs.data.template.enable_prot_template = False
    configs.data.pdb.template.enable_prot_template = False
    assert validate_static_graph_template_usage(configs) == {
        "pdb": False,
        "distillation": False,
    }


def test_fused_ordinary_adam_rejects_cpu_parameters():
    model = torch.nn.Linear(2, 1)

    with pytest.raises(
        ValueError,
        match=r"adam\.fused=true requires all trainable parameters to be CUDA",
    ):
        training_utils.get_optimizer(
            _ordinary_adam_config(fused=True),
            model,
        )


def test_non_fused_ordinary_adam_remains_standard_adam():
    model = torch.nn.Linear(2, 1)

    optimizer = training_utils.get_optimizer(
        _ordinary_adam_config(fused=False),
        model,
    )

    assert type(optimizer) is torch.optim.Adam
    assert optimizer.defaults["fused"] is None


def test_fused_validation_accepts_cuda_and_rejects_non_boolean():
    assert (
        training_utils._resolve_adam_fused(
            requested=True,
            parameter_device_types={"cuda"},
        )
        is True
    )
    with pytest.raises(TypeError, match=r"adam\.fused must be a bool"):
        training_utils._resolve_adam_fused(
            requested=1,
            parameter_device_types={"cuda"},
        )


def test_fused_true_is_forwarded_only_to_ordinary_adam(monkeypatch):
    model = torch.nn.Linear(2, 1)
    captured = {}
    sentinel = SimpleNamespace(param_groups=[])

    monkeypatch.setattr(
        training_utils,
        "_resolve_adam_fused",
        lambda **kwargs: True,
    )

    def capture_optimizer(
        params,
        optimizer_class,
        optimizer_sharding,
        **defaults,
    ):
        captured.update(
            {
                "optimizer_class": optimizer_class,
                "optimizer_sharding": optimizer_sharding,
                "defaults": defaults,
            }
        )
        return sentinel

    monkeypatch.setattr(training_utils, "_build_optimizer", capture_optimizer)

    optimizer = training_utils.get_optimizer(
        _ordinary_adam_config(fused=True),
        model,
    )

    assert optimizer is sentinel
    assert captured["optimizer_class"] is torch.optim.Adam
    assert captured["optimizer_sharding"] == "none"
    assert captured["defaults"]["fused"] is True


def test_run_manifest_audits_ddp_and_adam_fast_options():
    configs = SimpleNamespace(
        model_name="protenix-v2",
        seed=42,
        ddp=SimpleNamespace(gradient_as_bucket_view=True),
        adam=SimpleNamespace(use_adamw=False, fused=True),
        data=SimpleNamespace(train_sets=[], test_sets=[]),
    )

    manifest = build_run_manifest(
        configs,
        run_name="audit_fast_options",
        run_dir="/tmp/audit_fast_options",
    )

    assert manifest["distributed"]["ddp"] == {
        "gradient_as_bucket_view": True
    }
    assert manifest["training"]["adam"] == {
        "use_adamw": False,
        "fused": True,
    }
