# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import InitVar, dataclass
from typing import Literal, final

from torch import Tensor
from torch.nn import Module
from typing_extensions import override

from fairseq2.device import Device, SupportsDeviceTransfer
from fairseq2.nn.ops import CrossEntropy, cross_entropy
from fairseq2.nn.padding import PaddingMask


class SequenceModel(Module, ABC):
    """Represents a sequence model."""

    max_seq_len: int

    def __init__(self, max_seq_len: int) -> None:
        """
        :param max_seq_len: The maximum length of produced sequences.
        """
        super().__init__()

        self.max_seq_len = max_seq_len

    @abstractmethod
    def forward(self, batch: SequenceBatch) -> SequenceModelOutput:
        """
        :param batch: The batch of sequences to process.
        """


@dataclass
class SequenceBatch(SupportsDeviceTransfer):
    """Represents a sequence batch."""

    seqs: Tensor
    """The sequences. *Shape:* :math:`(N,S,*)`, where :math:`N` is the batch
    size, :math:`S` is the sequence length, and :math:`*` is any number of
    sequence-specific dimensions including none."""

    padding_mask: PaddingMask | None
    """The padding mask of :attr:`seqs`. *Shape:* :math:`(N,S)`, where :math:`N`
    is the batch size and :math:`S` is the sequence length."""

    target_mask: Tensor | None = None
    """The mask specifying the elements in ``seqs`` that should be treated as
    targets during model training or validation. *Shape:* :math:`(N,S)`, where
    :math:`N` is the batch size and :math:`S` is the sequence length."""

    example: object = None
    """The data example from which this batch was constructed."""

    @property
    def batch_size(self) -> int:
        """The size of the batch dimension."""
        return self.seqs.size(0)

    def num_elements(self) -> int:
        """Return the number of elements in the batch."""
        if self.padding_mask is None:
            return self.seqs.numel()

        return int(self.padding_mask.seq_lens.sum())

    def num_target_elements(self) -> int:
        """Return the number of target elements in the batch."""
        if self.target_mask is not None:
            return int(self.target_mask.sum())

        return self.num_elements()

    @override
    def to(self, device: Device) -> None:
        self.seqs = self.seqs.to(device)

        if self.padding_mask is not None:
            self.padding_mask = self.padding_mask.to(device)

        if self.target_mask is not None:
            self.target_mask = self.target_mask.to(device)


def as_auto_regressive_input(
    batch: SequenceBatch,
) -> tuple[SequenceBatch, SequenceBatch]:
    """Use ``batch`` to train an auto-regressive model.

    :returns:
        The tuple of input and target batches.
    """
    if (seq_len := batch.seqs.size(1)) < 2:
        raise ValueError(
            f"The sequence length of `batch.seqs` must be at least 2 for training, but is {seq_len} instead."
        )

    seqs, targets = batch.seqs[:, :-1], batch.seqs[:, 1:]

    if batch.padding_mask is None:
        padding_mask = None
    else:
        padding_mask = batch.padding_mask.trim(1)

    if batch.target_mask is None:
        seqs_target_mask, target_mask = None, None
    else:
        seqs_target_mask, target_mask = (
            batch.target_mask[:, :-1], batch.target_mask[:, 1:]  # fmt: skip
        )

    batch = SequenceBatch(seqs, padding_mask, seqs_target_mask, batch.example)

    target_batch = SequenceBatch(targets, padding_mask, target_mask)

    return batch, target_batch


@final
@dataclass
class SequenceModelOutput:
    """Holds the output of a sequence model."""

    logits: Tensor
    """The logits for next-step prediction. *Shape:* :math:`(N,S,T)`, where
    :math:`N` is the batch size, :math:`S` is the sequence length, and :math:`T`
    is the size of the vocabulary."""

    pad_idx: int | None
    """The index of the PAD symbols in the vocabulary."""

    loss_fn: InitVar[CrossEntropy | None] = None

    def __post_init__(self, loss_fn: CrossEntropy | None) -> None:
        self._loss_fn = loss_fn or cross_entropy

    def compute_loss(
        self,
        targets: Tensor,
        *,
        loss_mask: Tensor | None = None,
        reduction: Literal["sum", "mean"] = "sum",
        label_smoothing: float = 0.0,
        ignore_prefix_size: int = 0,
    ) -> Tensor:
        """
        Computes the negative log-likelihood loss.

        :param targets: The target indices. *Shape:* :math:`(N,S)`, where
            :math:`N` is the batch size and :math:`S` is the sequence length.
        :param loss_mask: The loss mask that specifies the elements in ``targets``
            that should be used in the loss computation. All non-masked elements
            will be ignored. *Shape:* Same as ``targets``.
        :param label_smoothing: The amount of label smoothing to apply while
            computing the loss.
        :param ignore_prefix_size: The number of steps from the beginning of the
            sequence that should be ignored in the loss computation.

        :returns: A scalar tensor representing the loss.
        """
        if ignore_prefix_size > 0:
            logits = self.logits[:, ignore_prefix_size:, :]
        else:
            logits = self.logits

        if ignore_prefix_size > 0:
            targets = targets[:, ignore_prefix_size:]

        # (N, S, T) -> (N x S, T)
        logits = logits.flatten(0, 1)

        # (N, S) -> (N x S)
        targets = targets.flatten(0, 1)

        # sum/mean: (), none: (N x S)
        loss = self._loss_fn(
            logits,
            targets,
            pad_idx=self.pad_idx,
            label_smoothing=label_smoothing,
            reduction=reduction if loss_mask is None else "none",
        )

        if loss_mask is None:
            return loss

        if ignore_prefix_size > 0:
            loss_mask = loss_mask[:, ignore_prefix_size:]

        # (N, S) -> (N x S)
        loss_mask = loss_mask.flatten(0, 1)

        loss = loss * loss_mask

        if reduction == "sum":
            return loss.sum()

        if reduction == "mean":
            return loss.mean()

        raise ValueError(
            f"`reduction` must be 'sum' or 'mean', but is '{reduction}' instead."
        )
