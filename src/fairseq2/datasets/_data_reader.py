# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator, Mapping
from typing import TypeVar, final

import torch
from typing_extensions import Self, override

from fairseq2.data import DataPipeline, DataPipelineError
from fairseq2.gang import Gang, GangError, all_sum
from fairseq2.logging import log
from fairseq2.utils.tensor import to_tensor

# isort: split

from fairseq2.datasets._config import DataReadOptions, SyncMode

BatchT_co = TypeVar("BatchT_co", covariant=True)


class DataReader(ABC, Iterator[list[BatchT_co]]):
    """Reads batches of examples from a dataset."""

    @abstractmethod
    def __iter__(self) -> Self: ...

    @abstractmethod
    def __next__(self) -> list[BatchT_co]: ...

    @abstractmethod
    def reset(self) -> None:
        """Reset state and move back to the first batch."""

    @abstractmethod
    def state_dict(self) -> dict[str, object]: ...

    @abstractmethod
    def load_state_dict(self, state_dict: Mapping[str, object]) -> None: ...

    @property
    @abstractmethod
    def dataset_name(self) -> str: ...

    @property
    @abstractmethod
    def split(self) -> str: ...

    @property
    @abstractmethod
    def num_accumulate(self) -> int:
        """The number of batches accumulated in each iteration."""


class DataReadError(Exception):
    dataset_name: str

    def __init__(self, dataset_name: str, split: str, message: str) -> None:
        super().__init__(message)

        self.dataset_name = dataset_name
        self.split = split


BatchT = TypeVar("BatchT")


@final
class DataPipelineReader(DataReader[BatchT]):
    """Reads batches of examples from a dataset using a :class:`DataPipeline`."""

    _dataset_name: str
    _split: str
    _pipeline: DataPipeline
    _pipeline_iter: Iterator[BatchT]
    _gang: Gang
    _options: DataReadOptions
    _eod: bool

    def __init__(
        self,
        dataset_name: str,
        split: str,
        pipeline: DataPipeline,
        gang: Gang,
        options: DataReadOptions,
        *,
        strict_state: bool = True,
    ) -> None:
        """
        :param name: The name of the dataset.
        :param pipeline: The data pipeline to iterate over.
        :param gang: The gang over which the underlying dataset is sharded.
        :param options: The read options.
        :param strict_state: If ``True``, the entire state of the data pipeline
            including shuffling and bucketing buffers will be included in the
            state dictionary.
        """
        self._dataset_name = dataset_name
        self._split = split
        self._pipeline = pipeline
        self._pipeline_iter = iter(pipeline)
        self._gang = gang
        self._options = options
        self._eod = False
        self._strict_state = strict_state

    @override
    def __iter__(self) -> Self:
        return self

    @override
    def __next__(self) -> list[BatchT]:
        if self._eod:
            raise StopIteration()

        batches = []

        num_accumulate = self._options.num_accumulate

        for idx in range(num_accumulate):
            try:
                batch = next(self._pipeline_iter)
            except StopIteration:
                break
            except DataPipelineError as ex:
                raise DataReadError(
                    self._dataset_name, self._split, f"The data pipeline has failed to read the next batch from the '{self._split}' split of the '{self._dataset_name}' dataset. See the nested exception for details."  # fmt: skip
                ) from ex

            batches.append(batch)

        # If we read less than `num_accumulate` batches, it means we reached end
        # of data.
        if self._options.drop_remainder and len(batches) != num_accumulate:
            batches.clear()

        local_num_batches = len(batches)

        if self._options.sync_batches and self._gang.size > 1:
            try:
                if self._options.sync_mode == SyncMode.UNTIL_LAST:
                    num_batches = _sum_num_batches(local_num_batches, self._gang)
                else:
                    num_batches = _min_num_batches(local_num_batches, self._gang)

                    if num_batches != local_num_batches:
                        batches = batches[:num_batches]
            except GangError as ex:
                raise DataReadError(
                    self._dataset_name, self._split, f"The batch synchronization of the gang processes has failed while reading the '{self._split}' split of the '{self._dataset_name}' dataset. See the nested exception for details."  # fmt: skip
                ) from ex
        else:
            num_batches = local_num_batches

        self._eod = num_batches == 0

        if self._eod:
            raise StopIteration()

        return batches

    @override
    def reset(self) -> None:
        self._eod = False

        self._pipeline.reset()

    @override
    def state_dict(self) -> dict[str, object]:
        return self._pipeline.state_dict(strict=self._strict_state)

    @override
    def load_state_dict(self, state_dict: Mapping[str, object]) -> None:
        self._eod = False

        self._pipeline.load_state_dict(state_dict)

    @property
    @override
    def dataset_name(self) -> str:
        return self._dataset_name

    @property
    @override
    def split(self) -> str:
        return self._split

    @property
    @override
    def num_accumulate(self) -> int:
        return self._options.num_accumulate


def _min_num_batches(num_batches: int, gang: Gang) -> int:
    all_num_batches = torch.zeros((gang.size,), device=gang.device, dtype=torch.int64)

    num_batches_pt = to_tensor([num_batches], device=gang.device)

    gang.all_gather(all_num_batches, num_batches_pt)

    min_num_batches = int(all_num_batches.min())
    if min_num_batches != 0:
        return min_num_batches

    # If not all processes have reached end of data, report the ones that have
    # reached for debugging purposes.
    if log.is_enabled_for_debug() and all_num_batches.sum() > 0:
        ranks = all_num_batches.bool().logical_not_().nonzero().squeeze(-1).tolist()

        s = ", ".join(str(r) for r in ranks)

        log.debug("End of data reached at rank(s) {}.", s)

    return 0


def _sum_num_batches(num_batches: int, gang: Gang) -> int:
    total_num_batches = all_sum(gang, num_batches)

    return int(total_num_batches)
