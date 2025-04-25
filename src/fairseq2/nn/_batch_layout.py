# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from collections.abc import Sequence
from typing import final

import torch
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence

from fairseq2.device import CPU, Device


#@final
#class PackedBatchLayout(BatchLayout):
#    def __init__(
#        self,
#        shape: tuple[int, int],
#        seq_lens: list[int],
#    ) -> None:


@final
class BatchLayout:
    _shape: tuple[int, int]
    _seq_lens: list[int]
    _seq_lens_pt: Tensor | None
    _num_elements: int
    _max_seq_len: int
    _is_packed: bool
    _is_padded: bool
    _padding_mask: Tensor | None
    _device: Device

    def __init__(
        self,
        shape: tuple[int, int],
        seq_lens: list[int] | None,
        device: Device,
        *,
        is_packed: bool = False,
    ) -> None:
        self._shape = shape

        batch_size, batch_width = shape

        if seq_lens is None:
#            if is_packed and batch_size != 1:
#                raise ValueError(
#                    f"The batch size must be 1 for a packed batch, but is {batch_size} instead."
#                )

            self._seq_lens = [batch_width] * batch_size

            self._num_elements = batch_width * batch_size

            self._is_padded = False

            self._max_seq_len = batch_width
        else:
            self._max_seq_len = 0

            if not seq_lens:
                raise ValueError("`seq_lens` must not be empty.")

            if is_packed:
                self._num_elements = 0

                for idx, seq_len in enumerate(seq_lens):
                    if seq_len < 0:
                        raise ValueError(
                            f"`seq_lens[{idx}]` must be greater than or equal to 0."
                        )

                    self._num_elements += seq_len

                    self._max_seq_len = max(self._max_seq_len, seq_len)

                sz = batch_size * batch_width
#                if self._num_elements > batch_width:
                if self._num_elements > sz:
                    raise ValueError(
                        "dede"
#                        f"`sum(seq_lens)` must be less than or equal to the batch width ({batch_width}) when the batch is packed, but is {self._num_elements} instead."
                    )

                self._is_padded = self._num_elements < sz
            else:
                if len(seq_lens) != batch_size:
                    raise ValueError(
                        f"`len(seq_lens)` must be equal to the batch size ({batch_size}) when the batch is non-packed, but is {len(seq_lens)} instead."
                    )

                self._is_padded = False

                self._num_elements = 0

                for idx, seq_len in enumerate(seq_lens):
                    if seq_len < 0:
                        raise ValueError(
                            f"`seq_lens[{idx}]` must be greater than or equal to 0."
                        )

                    if seq_len > batch_width:
                        raise ValueError(
                            f"`seq_lens[{idx}]` must be less than or equal to the batch width ({batch_width}) when the batch is non-packed, but is {seq_len} instead."
                        )

                    self._max_seq_len = max(self._max_seq_len, seq_len)

                    if seq_len < batch_width:
                        self._is_padded = True

                    self._num_elements += seq_len

            self._seq_lens = seq_lens

        self._seq_lens_pt = None

        self._is_packed = is_packed

        self._padding_mask = None

        self._device = device

        if self._is_packed:
            indices = []

            for seq_len in self._seq_lens:
                indices.append(torch.arange(seq_len, device=self._device))

            self._pos_indices = torch.cat(indices)

            self._pos_indices = self._pos_indices.unsqueeze(0)

            sub = torch.arange(batch_size, device=self._device)
            sub = sub.repeat_interleave(batch_width)

            self._pos_indices -= sub

            self._pos_indices = self._pos_indices.view(batch_size, batch_width)
        else:
            batch_size, batch_width = self._shape

            indices = torch.arange(batch_width, device=self._device)

            # (N) -> (N, S)
            self._pos_indices = indices.expand(batch_size, -1)

    @staticmethod
    def of(
        batch: Tensor, seq_lens: list[int] | None = None, is_packed: bool = False
    ) -> BatchLayout:
        if batch.ndim < 2:
            raise ValueError(
                f"`batch` must be at least two dimensional, but has {batch.ndim} dimension instead."
            )

        batch_size, batch_width = batch.shape[:2]

        return BatchLayout(
            (batch_size, batch_width), seq_lens, batch.device, is_packed=is_packed
        )

    def trim(self) -> BatchLayout:
        batch_size, batch_width = self._shape

        if self._is_packed:
            seq_lens = self._seq_lens.copy()

            seq_lens[-1] = max(seq_lens[-1] - 1, 0)
        else:
            seq_lens = [max(seq_len - 1, 0) for seq_len in self._seq_lens]

        shape = (batch_size, batch_width - 1)

        return BatchLayout(shape, seq_lens, self._device, is_packed=self._is_packed)

    def to(self, device: Device) -> BatchLayout:
        return BatchLayout(
            self._shape, self._seq_lens, device, is_packed=self._is_packed
        )

    def get_seq_ranges(self) -> list[tuple[int, int]]:
        if self._is_packed:
            seq_ranges = []

            offset = 0

            for seq_len in self._seq_lens:
                seq_ranges.append((offset, seq_len))

                offset += seq_len

            return seq_ranges
        else:
            return [(0, seq_len) for seq_len in self._seq_lens]

    def get_padding_mask(self) -> Tensor:
        batch_size, batch_width = self._shape

        if self._padding_mask is None:
            if self._is_packed:
                # (N, S)
                self._padding_mask = torch.zeros(
                    (batch_size * batch_width), device=self._device, dtype=torch.bool
                )

                self._padding_mask[: self._num_elements] = True

                self._padding_mask = self._padding_mask.view(batch_size, batch_width)
            else:
                # (S)
                indices = torch.arange(batch_width, device=self._device)

                # (N) -> (N, S)
                indices = indices.expand(batch_size, -1)

                # (N) -> (N, S)
                lengths = self.seq_lens_pt.unsqueeze(1).expand(-1, batch_width)

                self._padding_mask = indices < lengths

        return self._padding_mask

    def get_position_indices(self) -> Tensor:
#        if self._pos_indices is None:
#            if self._is_packed:
#                indices = []
#
#                for seq_len in self._seq_lens:
#                    indices.append(torch.arange(seq_len, device=self._device))
#
#                self._pos_indices = torch.cat(indices)
#
#                self._pos_indices = self._pos_indices.unsqueeze(0)
#            else:
#                batch_size, batch_width = self._shape
#
#                indices = torch.arange(batch_width, device=self._device)
#
#                # (N) -> (N, S)
#                self._pos_indices = indices.expand(batch_size, -1)

        return self._pos_indices

    def __repr__(self) -> str:
        s = (
            f"shape={self._shape}, "
            f"seq_lens={self._seq_lens}, "
            f"num_elements={self._num_elements}, "
            f"is_packed={self._is_packed}, "
            f"is_padded={self._is_padded}"
        )

        return f"BatchLayout({s})"

    @property
    def shape(self) -> tuple[int, int]:
        return self._shape

    @property
    def seq_lens(self) -> Sequence[int]:
        return self._seq_lens

    @property
    def seq_lens_pt(self) -> Tensor:
        if self._seq_lens_pt is None:
            if self._is_packed or self._is_padded:
                if self._device.type == "cuda":
                    seq_lens_pt = torch.tensor(
                        self._seq_lens, device=CPU, pin_memory=True
                    )

                    # Avoid host-to-device sync.
                    self._seq_lens_pt = seq_lens_pt.to(
                        device=self._device, non_blocking=True
                    )
                else:
                    self._seq_lens_pt = torch.tensor(
                        self._seq_lens, device=self._device
                    )
            else:
                batch_size, batch_width = self._shape

                self._seq_lens_pt = torch.full(
                    (batch_width,), batch_size, device=self._device
                )

        return self._seq_lens_pt

    @property
    def num_elements(self) -> int:
        return self._num_elements

    @property
    def max_seq_len(self) -> int:
        return self._max_seq_len

    @property
    def is_packed(self) -> bool:
        return self._is_packed

    @property
    def is_padded(self) -> bool:
        return self._is_padded
