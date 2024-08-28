# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from fairseq2.recipes.cli import Cli, RecipeCommandHandler
from fairseq2.recipes.utils.sweep import default_sweep_tagger
from fairseq2.recipes.w2vbert.train import load_w2vbert_trainer, w2vbert_train_presets


def _setup_w2vbert_cli(cli: Cli) -> None:
    default_sweep_tagger.extend_allow_set(
        "max_audio_len", "min_audio_len", "normalize_audio"
    )

    group = cli.add_group("w2vbert", help="w2v-BERT pretraining recipes")

    # Train
    train_handler = RecipeCommandHandler(
        loader=load_w2vbert_trainer,
        preset_configs=w2vbert_train_presets,
        default_preset="arrival-test",
    )

    group.add_command(
        name="train",
        handler=train_handler,
        help="train a w2v-BERT model",
    )
