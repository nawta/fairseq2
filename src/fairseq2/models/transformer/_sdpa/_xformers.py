# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import Any, final

from torch import Tensor
from typing_extensions import override

from fairseq2.device import Device
from fairseq2.error import InternalError, NotSupportedError
from fairseq2.models.transformer._attention_bias import (
    AttentionBias,
    AttentionBiasCache,
    CausalAttentionBias,
    IdentityBias,
)
from fairseq2.models.transformer._sdpa._base import SDPA
from fairseq2.nn import BatchLayout


@final
class xFormersSDPA(SDPA):
    dropout_p: float

    def __init__(self, bias: AttentionBias, *, dropout_p: float = 0.0) -> None:
        """
        :param dropout_p: The dropout probability on attention weights.
        """
        super().__init__(bias)

        self.dropout_p = dropout_p

    @override
    def forward(
        self,
        seqs: Tensor,
        seqs_layout: BatchLayout,
        keys: Tensor,
        keys_layout: BatchLayout,
        values: Tensor,
        bias_cache: AttentionBiasCache,
        *,
        needs_weights: bool = False,
    ) -> tuple[Tensor, Tensor | None]:
        from xformers.ops.fmha import memory_efficient_attention

        if needs_weights:
            raise NotSupportedError("`FlexSDPA` does not support `needs_weights`.")

        if seqs_layout.is_packed ^ keys_layout.is_packed:
            raise ValueError(
                "`seqs_layout` and `keys_layout` must be both packed or non-packed."
            )

        bias = self._maybe_get_attention_bias(
            seqs_layout, keys_layout, seqs.device, bias_cache
        )

        if not self.training:
            dropout_p = 0.0
        else:
            dropout_p = self.dropout_p

        #        bsz = seqs.size(0)

        # (N, H, S, K) -> (N, S, H, K)
        seqs = seqs.transpose(1, 2)

        # (N, H, S_kv, K) -> (N, S_kv, H, K)
        keys = keys.transpose(1, 2)

        # (N, H, S_kv, V) -> (N, S_kv, H, V)
        values = values.transpose(1, 2)

        #        seqs = seqs.flatten(0, 1).unsqueeze(0)
        #        keys = keys.flatten(0, 1).unsqueeze(0)
        #        values = values.flatten(0, 1).unsqueeze(0)

        attns = memory_efficient_attention(
            seqs, keys, values, attn_bias=bias, p=dropout_p
        )

        #        attns = attns.squeeze(0).unflatten(0, (bsz, -1))

        # (N, S, H, V) -> (N, H, S, V)
        attns = attns.transpose(1, 2)

        return attns, None

    def _maybe_get_attention_bias(
        self,
        seqs_layout: BatchLayout,
        keys_layout: BatchLayout,
        device: Device,
        bias_cache: AttentionBiasCache,
    ) -> Any:
        from xformers.ops.fmha.attn_bias import AttentionBias as FmhaAttentionBias
        from xformers.ops.fmha.attn_bias import (
            BlockDiagonalCausalLocalAttentionPaddedKeysMask,
            BlockDiagonalCausalWithOffsetPaddedKeysMask,
            BlockDiagonalMask,
            BlockDiagonalPaddedKeysMask,
            LowerTriangularFromBottomRightMask,
        )

        full_seqs = not seqs_layout.is_packed and not seqs_layout.is_padded
        full_keys = not keys_layout.is_packed and not keys_layout.is_padded

        if full_seqs and full_keys:
            if isinstance(self.bias, IdentityBias):
                return None

            if isinstance(self.bias, CausalAttentionBias):
                bias = LowerTriangularFromBottomRightMask()

                attn_window_len = self.bias.attn_window_len
                if attn_window_len is not None:
                    bias = bias.make_local_attention(attn_window_len)

                return bias

            raise NotSupportedError(
                f"`xFormersSDPA` supports not support `{self.bias}`."
            )

        bias = bias_cache.maybe_get(self.bias, impl="xformers", kls=FmhaAttentionBias)

        if bias is not None:
            return bias

        if seqs_layout.is_packed:
            if not keys_layout.is_packed:
                raise InternalError("dede")

            if isinstance(self.bias, IdentityBias):
                bias = BlockDiagonalMask.from_seqlens(
                    seqs_layout.seq_lens, keys_layout.seq_lens, device=device
                )

            if isinstance(self.bias, CausalAttentionBias):
                bias = BlockDiagonalMask.from_seqlens(
                    seqs_layout.seq_lens, keys_layout.seq_lens, device=device
                )

                attn_window_len = self.bias.attn_window_len
                if attn_window_len is None:
                    bias = bias.make_causal_from_bottomright()
                else:
                    bias = bias.make_local_attention_from_bottomright()

            raise NotSupportedError(
                f"`xFormersSDPA` supports not support `{self.bias}`."
            )
        #        bias = None
        #        if keys_layout.is_padded:
        #            keys_len = keys_layout.shape[1]
        #
        #            if isinstance(self.bias, IdentityBias):
        #                bias = BlockDiagonalPaddedKeysMask.from_seqlens(
        #                    seqs_layout.seq_lens,
        #                    keys_len,
        #                    keys_layout.seq_lens,
        #                    device=device,
        #                )
        #            elif isinstance(self.bias, CausalAttentionBias):
        #                attn_window_len = self.bias.attn_window_len
        #                if attn_window_len is None:
        #                    bias = BlockDiagonalCausalWithOffsetPaddedKeysMask.from_seqlens(
        #                        seqs_layout.seq_lens,
        #                        keys_len,
        #                        keys_layout.seq_lens,
        #                        device=device,
        #                    )
        #                else:
        #                    bias = BlockDiagonalCausalLocalAttentionPaddedKeysMask.from_seq_lens_local(
        #                        seqs_layout.seq_lens,
        #                        keys_len,
        #                        keys_layout.seq_lens,
        #                        window_size=attn_window_len,
        #                    )
        #        bias = None
        #        if keys_layout.is_padded:
        #            seq_lens = []
        #            for s in seqs_layout.seq_lens:
        #                seq_lens.append(s)
        #                padding = seqs_layout.shape[1] - s
        #                if padding > 0:
        #                    seq_lens.append(padding)
        #            key_lens = []
        #            for s in keys_layout.seq_lens:
        #                key_lens.append(s)
        #                padding = keys_layout.shape[1] - s
        #                if padding > 0:
        #                    key_lens.append(padding)
        #
        #            keys_len = keys_layout.shape[1]
        #
        #            if isinstance(self.bias, IdentityBias):
        #                bias = BlockDiagonalMask.from_seqlens(seq_lens, key_lens, device=device)
        #            elif isinstance(self.bias, CausalAttentionBias):
        #                bias = BlockDiagonalMask.from_seqlens(seq_lens, key_lens, device=device)
        #
        #                attn_window_len = self.bias.attn_window_len
        #                if attn_window_len is None:
        #                    bias = bias.make_causal_from_bottomright()
        #                else:
        #                    bias = bias.make_local_attention_from_bottomright(attn_window_len)

        bias_cache.set(self.bias, impl="xformers", value=bias)

        return bias

    def extra_repr(self) -> str:
        """:meta private:"""
        s = super().extra_repr()

        return f"{s}, dropout_p={self.dropout_p:G}"
