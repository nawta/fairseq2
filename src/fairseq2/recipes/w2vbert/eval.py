# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import final

import torch
from torch.nn import Module
from typing_extensions import override

from fairseq2.assets import AssetNotFoundError, default_asset_store
from fairseq2.checkpoint import CheckpointModelMetadataProvider
from fairseq2.config_registry import ConfigRegistry
from fairseq2.datasets.batching import LengthBatching
from fairseq2.datasets.speech import GenericSpeechDataset, load_speech_dataset
from fairseq2.gang import Gang
from fairseq2.logging import get_log_writer
from fairseq2.models import load_model
from fairseq2.models.sequence import SequenceBatch
from fairseq2.models.w2vbert import W2VBertModel, W2VBertOutput
from fairseq2.nn.utils.module import remove_parametrizations
from fairseq2.recipes.evaluator import AbstractEvalUnit, Evaluator
from fairseq2.recipes.utils.asset import (
    AssetReference,
    asset_as_path,
    retrieve_asset_card,
)
from fairseq2.recipes.utils.log import log_model
from fairseq2.recipes.utils.setup import (
    broadcast_model,
    check_model_type,
    setup_root_gang,
)
from fairseq2.recipes.w2vbert.common import W2VBertMetricBag
from fairseq2.typing import META, DataType
from fairseq2.utils.profiler import Stopwatch

log = get_log_writer(__name__)


@dataclass(kw_only=True)
class W2VBertEvalConfig:
    """Holds the configuration of a w2v-BERT model evaluation task."""

    # Data
    dataset: AssetReference = "librispeech_960h"
    """The name, path or path to the asset card of the dataset to evaluate on."""

    split: str = "valid"
    """The name of the eval data split."""

    min_audio_len: int = 32_000
    """The minimum audio sequence length."""

    max_audio_len: int = 250_000
    """The maximum audio sequence length."""

    max_num_elements: int = 1_500_000
    """The maximum number of elements per batch."""

    normalize_audio: bool = False
    """If ``True``, normalizes audio to have zero mean and unit variance."""

    num_prefetch: int = 4
    """The number of batches to prefetch in background."""

    # Model
    model: AssetReference = "foo"
    """The name or path to the asset card of the w2v-BERT model to evaluate."""

    checkpoint_dir: Path | None = None
    """The checkpoint directory containing models saved by :class:`FileCheckpointManager`."""

    dtype: DataType = torch.float16
    """The data type of the model."""

    # Loss
    bert_loss_weight: float = 1.0
    """The weight of masked prediction loss in loss computation."""

    w2v2_loss_weight: float = 1.0
    """The weight of wav2vec 2.0 loss in loss computation."""

    diversity_loss_weight: float = 0.1
    """The weight of the diversity loss."""

    feature_penalty_weight: float = 0.0
    """The weight of the regularization penalty applied to the extracted features."""

    bert_label_smoothing: float = 0.0
    """The amount of label smoothing when computing masked prediction loss."""

    # Misc
    seed: int = 2
    """The random number generator seed to use."""


w2vbert_eval_presets = ConfigRegistry[W2VBertEvalConfig]()

w2vbert_eval_preset = w2vbert_eval_presets.decorator


@w2vbert_eval_preset("300m")
def _300m() -> W2VBertEvalConfig:
    return W2VBertEvalConfig()


def load_w2vbert_evaluator(
    config: W2VBertEvalConfig, output_dir: Path
) -> Evaluator[SequenceBatch]:
    """Load an :class:`Evaluator` for w2v-BERT model evaluation."""
    wall_watch = Stopwatch(start=True)

    if config.checkpoint_dir is not None:
        default_asset_store.metadata_providers.append(
            CheckpointModelMetadataProvider(
                config.checkpoint_dir, lower_score_better=True
            )
        )

    gang = setup_root_gang(log)

    # Load the dataset.
    try:
        dataset_card = retrieve_asset_card(config.dataset)
    except AssetNotFoundError:
        dataset_card = None

    if dataset_card is not None:
        log.info("Loading {} speech dataset.", dataset_card.name)

        dataset = load_speech_dataset(dataset_card)

        log.info("Dataset loaded.")
    else:
        dataset_path = asset_as_path(config.dataset)

        dataset = GenericSpeechDataset.from_path(dataset_path)

    model_card = retrieve_asset_card(config.model)

    # Load the model.
    log.info("Loading {} model on rank 0.", model_card.name)

    if gang.rank == 0:
        init_device = gang.device
    else:
        init_device = META

    try:
        model = load_model(model_card, device=init_device, dtype=config.dtype)
    except ValueError as ex:
        raise ValueError(
            "The model cannot be initialized. See nested exception for details."
        ) from ex

    if not isinstance(model, W2VBertModel):
        raise ValueError(
            f"The model must be of type `{W2VBertModel}`, but is of type `{type(model)}` instead."
        )

    gang.barrier()

    log.info("Model loaded on rank 0.")

    remove_parametrizations(model)

    # Distribute the model to all processes in the gang.
    if gang.size != 1:
        broadcast_model(model, gang, log)

    log_model(model, log)

    # Initialize the evaluation unit.
    unit = W2VBertEvalUnit(
        model,
        gang,
        config.bert_loss_weight,
        config.w2v2_loss_weight,
        config.diversity_loss_weight,
        config.feature_penalty_weight,
        config.bert_label_smoothing,
    )

    seed = config.seed

    try:
        data_reader = dataset.create_reader(
            config.split,
            gang,
            config.max_audio_len,
            batching=LengthBatching(config.max_num_elements),
            dtype=config.dtype,
            min_audio_len=config.min_audio_len,
            normalize_audio=config.normalize_audio,
            num_prefetch=config.num_prefetch,
            seed=seed,
        )
    except ValueError as ex:
        raise ValueError(
            "The data reader cannot be initialized. See nested exception for details."
        ) from ex

    seed += 1

    # Initialize the evaluator.
    return Evaluator[SequenceBatch](
        units=[unit],
        data_readers=[data_reader],
        root_gang=gang,
        tb_dir=output_dir.joinpath("tb"),
        metrics_dir=output_dir.joinpath("metrics"),
        seed=seed,
        wall_watch=wall_watch,
    )


@final
class W2VBertEvalUnit(AbstractEvalUnit[SequenceBatch]):
    """Represents a w2v-BERT model evaluation unit."""

    _bert_loss_weight: float
    _w2v2_loss_weight: float
    _diversity_loss_weight: float
    _feature_penalty_weight: float
    _metric_bag: W2VBertMetricBag

    def __init__(
        self,
        model: Module,
        gang: Gang,
        bert_loss_weight: float,
        w2v2_loss_weight: float,
        diversity_loss_weight: float,
        feature_penalty_weight: float,
        bert_label_smoothing: float,
    ) -> None:
        """
        :param model:
            The w2v-BERT model. Might be wrapped with DDP or FSDP.
        :param gang:
            The gang for distributed evaluation.
        :param bert_loss_weight:
            The weight of masked prediction loss in loss computation.
        :param bert_label_smoothing:
            The amount of label smoothing when computing masked prediction loss.
        :param diversity_loss_weight:
            The weight of diversity in loss computation.
        :param feature_penalty_weight:
            The weight of the feature penalty in loss computation.
        :param bert_label_smoothing:
            The amount of label smoothing when computing masked prediction loss.
        """
        super().__init__(model)

        check_model_type(model, W2VBertModel)

        self._bert_loss_weight = bert_loss_weight
        self._w2v2_loss_weight = w2v2_loss_weight
        self._diversity_loss_weight = diversity_loss_weight
        self._feature_penalty_weight = feature_penalty_weight
        self._bert_label_smoothing = bert_label_smoothing

        self._metric_bag = W2VBertMetricBag(gang)

    @override
    def __call__(self, batch: SequenceBatch) -> None:
        output = self._forward(batch)

        loss = output.compute_loss(
            bert_loss_weight=self._bert_loss_weight,
            w2v2_loss_weight=self._w2v2_loss_weight,
            diversity_loss_weight=self._diversity_loss_weight,
            feature_penalty_weight=self._feature_penalty_weight,
            bert_label_smoothing=self._bert_label_smoothing,
        )

        batch_size, seq_len = output.w2v2_output.logits.shape[:2]

        num_targets = batch_size * seq_len

        self._metric_bag.update_losses(loss, num_targets)

        self._metric_bag.update_accuracy(output)

        self._metric_bag.update_quantizer_metrics(output.w2v2_output.quantizer_output)

        self._metric_bag.update_batch_metrics(batch)

    def _forward(self, batch: SequenceBatch) -> W2VBertOutput:
        return self._model(batch)  # type: ignore[no-any-return]

    @property
    @override
    def metric_bag(self) -> W2VBertMetricBag:
        return self._metric_bag
