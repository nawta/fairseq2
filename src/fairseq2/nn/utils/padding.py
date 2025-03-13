# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from collections.abc import Sequence

from torch import Tensor

from fairseq2.device import Device
from fairseq2.nn import BatchLayout


def apply_mask(
    seqs: Tensor, mask: Tensor, *, fill_value: int | float | Tensor = 0
) -> Tensor:
    """
    Applies the specified mask to ``seqs``.

    :param seqs: The sequences to mask. *Shape:* :math:`(N,S,*)`, where :math:`N`
        is the batch size, :math:`S` is the sequence length, and :math:`*` is
        any number of sequence-specific dimensions including none.
    :param mask: The boolean mask.

    :returns: The input sequences with mask applied. *Shape:* Same as ``seqs``.
    """
    for _ in range(seqs.ndim - mask.ndim):
        mask = mask.unsqueeze(-1)

    return seqs.where(mask, fill_value)


def pad_seqs(
    seqs: Sequence[Tensor],
    pad_value: int = 0,
    pad_to_multiple: int = 1,
    device: Device | None = None,
) -> tuple[Tensor, BatchLayout]:
    """
    Stacks ``seqs`` along a new batch dimension and pad them to equal length.

    :param seqs: The list of variable length sequences. All elements in ``seqs``
        are expected to have the same shape except the first dimension.
    :param pad_value: The value for padded positions.
    :param pad_to_multiple: The sequence dimension is rounded up to the nearest
        multiple of the specified value.

    :returns:
        - The padded sequence stack. *Shape:* :math:`(N,S,*)`, where :math:`N`
          is the batch size, :math:`S` is the sequence length, and :math:`*` is
          any number of sequence-specific dimensions including none.
    """
    from fairseq2n.bindings.data.data_pipeline import (  # type: ignore[import-not-found]
        Collater,
    )

    data = Collater(pad_value=pad_value, pad_to_multiple=pad_to_multiple)(seqs)

    padded_seqs = data["seqs"]

    if device is not None:
        padded_seqs = padded_seqs.to(device)

    seq_lens = data["seq_lens"]
    # TODO: fix!
    seq_lens_ = seq_lens.tolist()
    seqs_layout = BatchLayout.of(padded_seqs, seq_lens_)

    return padded_seqs, seqs_layout
