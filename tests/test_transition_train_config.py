from copy import deepcopy

from configs.configs_base import configs as configs_base
from protenix.config.config import parse_configs


def test_pairformer_transition_training_options_parse_from_cli():
    configs = parse_configs(
        configs=deepcopy(configs_base),
        arg_str=(
            "--model.pairformer.blocks_per_ckpt 2 "
            "--model.pairformer.train_pair_transition_chunk_size 128 "
            "--model.pairformer.train_pair_transition_checkpoint_chunks true"
        ),
        fill_required_with_null=True,
    )

    assert configs.model.pairformer.blocks_per_ckpt == 2
    assert configs.model.pairformer.train_pair_transition_chunk_size == 128
    assert configs.model.pairformer.train_pair_transition_checkpoint_chunks is True


def test_diffusion_transition_training_options_parse_from_cli():
    configs = parse_configs(
        configs=deepcopy(configs_base),
        arg_str=(
            "--model.diffusion_module.train_pair_transition_chunk_size 128 "
            "--model.diffusion_module.train_pair_transition_checkpoint_chunks true"
        ),
        fill_required_with_null=True,
    )

    assert configs.model.diffusion_module.train_pair_transition_chunk_size == 128
    assert (
        configs.model.diffusion_module.train_pair_transition_checkpoint_chunks is True
    )
