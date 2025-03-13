# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from torch import Tensor

from fairseq2.data import SequenceData
from fairseq2.device import Device
from fairseq2.nn import BatchLayout


def get_seqs_with_layout(
    data: SequenceData, device: Device | None = None
) -> tuple[Tensor, BatchLayout]:
    """Returns the sequences along with their layout from ``data``."""
    seqs = data["seqs"]

    if device is not None:
        seqs = seqs.to(device)

    seq_lens = data["seq_lens"]

    # TODO: fix!
    seq_lens_ = seq_lens.tolist()
    #    if device is not None:
    #        seq_lens = seq_lens.to(device)

    seqs_layout = BatchLayout.of(seqs, seq_lens_)

    return seqs, seqs_layout
