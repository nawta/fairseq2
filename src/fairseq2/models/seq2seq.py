# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from torch import Tensor
from torch.nn import Module
from typing_extensions import override

from fairseq2.device import Device, SupportsDeviceTransfer
from fairseq2.models.sequence import SequenceBatch, SequenceModelOutput
from fairseq2.nn import BatchLayout


class Seq2SeqModel(Module, ABC):
    """Represents a sequence-to-sequence model."""

    max_source_seq_len: int
    max_target_seq_len: int

    def __init__(self, max_source_seq_len: int, max_target_seq_len: int) -> None:
        """
        :param max_target_seq_len: The maximum length of produced sequences.
        """
        super().__init__()

        self.max_source_seq_len = max_source_seq_len
        self.max_target_seq_len = max_target_seq_len

    @abstractmethod
    def forward(self, batch: Seq2SeqBatch) -> SequenceModelOutput:
        """
        :param batch: The batch of sequences to process.
        """


@dataclass
class Seq2SeqBatch(SupportsDeviceTransfer):
    """Represents a sequence-to-sequence batch."""

    source_seqs: Tensor
    """The source sequences. *Shape:* :math:`(N,S_{src},*)`, where :math:`N` is
    the batch size, :math:`S_{src}` is the source sequence length, and :math:`*`
    is any number of sequence-specific dimensions including none."""

    source_seqs_layout: BatchLayout

    target_seqs: Tensor
    """The target sequences. *Shape:* :math:`(N,S_{tgt},*)`, where :math:`N` is
    the batch size, :math:`S_{tgt}` is the target sequence length, and :math:`*`
    is any number of sequence-specific dimensions including none."""

    target_seqs_layout: BatchLayout

    example: object = None
    """The data example from which this batch was constructed."""

    @property
    def batch_size(self) -> int:
        """The size of the batch dimension."""
        return self.target_seqs.size(0)

    @property
    def num_source_elements(self) -> int:
        """The number of source elements in the batch."""
        return self.source_seqs_layout.num_elements

    @property
    def num_target_elements(self) -> int:
        """The number of target elements in the batch."""
        return self.target_seqs_layout.num_elements

    @override
    def to(self, device: Device) -> None:
        """Moves the batch to ``device``."""
        self.source_seqs = self.source_seqs.to(device)
        self.source_seqs_layout = self.source_seqs_layout.to(device)

        self.target_seqs = self.target_seqs.to(device)
        self.target_seqs_layout = self.target_seqs_layout.to(device)


def as_auto_regressive_input(batch: Seq2SeqBatch) -> tuple[Seq2SeqBatch, SequenceBatch]:
    """Use ``batch`` to train an auto-regressive model.

    :returns:
        The tuple of input and target batches.
    """
    for idx, seq_len in enumerate(batch.target_seqs_layout.seq_lens):
        if seq_len < 2:
            raise ValueError(
                f"The length of `batch.target_seqs[{idx}]` must be greater than or equal to 2 for training, but is {seq_len} instead."
            )

    seqs = batch.target_seqs[:, :-1]

    seqs_layout = batch.target_seqs_layout.trim()

    input_batch = Seq2SeqBatch(
        batch.source_seqs,
        batch.source_seqs_layout,
        seqs,
        seqs_layout,
        batch.example,
    )

    targets = batch.target_seqs[:, 1:]

    target_batch = SequenceBatch(targets, seqs_layout)

    return input_batch, target_batch
