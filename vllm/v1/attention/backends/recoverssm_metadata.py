# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import abc
from collections.abc import Sequence
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class RecoverSSMPostprocessMetadata:
    """Metadata used during postprocessing for align-mode prefix caching."""

    num_spec_decodes: int
    request_indices: torch.Tensor | None
    block_table: torch.Tensor
    num_computed_tokens: torch.Tensor
    block_size: int


class RecoverSSMMetadata(abc.ABC):
    @classmethod
    @abc.abstractmethod
    def commit_recoverssm_state(
        cls,
        metadata: Sequence["RecoverSSMMetadata"],
        num_accepted_tokens: torch.Tensor,
    ) -> RecoverSSMPostprocessMetadata | None:
        """Restore accepted state for all compatible attention groups."""
        raise NotImplementedError
