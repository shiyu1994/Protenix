import importlib.util
import os
from pathlib import Path
from unittest import mock

import pytest


def _load_compile_module():
    path = (
        Path(__file__).resolve().parents[1]
        / "protenix/model/layer_norm/torch_ext_compile.py"
    )
    spec = importlib.util.spec_from_file_location(
        "protenix_layer_norm_compile_test",
        path,
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_cuda_arch_list_uses_only_visible_device_capabilities():
    module = _load_compile_module()
    with mock.patch.dict(os.environ, {}, clear=False), mock.patch.object(
        module.torch.cuda, "is_available", return_value=True
    ), mock.patch.object(
        module.torch.cuda, "device_count", return_value=3
    ), mock.patch.object(
        module.torch.cuda,
        "get_device_capability",
        side_effect=[(9, 0), (8, 0), (9, 0)],
    ), mock.patch.object(
        module.subprocess,
        "check_output",
        return_value="compute_80\ncompute_90\n",
    ):
        os.environ.pop("PROTENIX_CUDA_ARCH_LIST", None)
        assert module._resolve_cuda_arch_list() == "8.0;9.0"


def test_cuda_arch_override_rejects_architecture_unsupported_by_nvcc():
    module = _load_compile_module()
    with mock.patch.dict(
        os.environ,
        {"PROTENIX_CUDA_ARCH_LIST": "10.0"},
        clear=False,
    ), mock.patch.object(
        module.subprocess,
        "check_output",
        return_value="compute_80\ncompute_90\n",
    ):
        with pytest.raises(RuntimeError, match="compute_100"):
            module._resolve_cuda_arch_list()
