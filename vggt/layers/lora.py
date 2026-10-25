"""LoRA wrappers for VGGT linear layers (added for ReVA-HPE)."""

from typing import Iterable, Optional

import torch
import torch.nn as nn

from wcmatch import fnmatch

_GLOB_FLAGS = (
    fnmatch.CASE
    | fnmatch.DOTMATCH
    | fnmatch.EXTMATCH
    | fnmatch.SPLIT
)


class LoRALinear(nn.Module):
    """
    LoRA wrapper for a Linear layer.

    Adds a low-rank adapter: W(x) + scale * B(A(x)).
    The base Linear weights are frozen by default.
    """

    def __init__(self, base: nn.Linear, rank: int, alpha: int, dropout: float = 0.0):
        super().__init__()
        if rank <= 0:
            raise ValueError(f"rank must be positive, got {rank}")
        if alpha <= 0:
            raise ValueError(f"alpha must be positive, got {alpha}")

        self.base = base
        self.rank = rank
        self.alpha = alpha
        self.scale = alpha / float(rank)
        self.dropout = nn.Dropout(dropout)

        self.lora_A = nn.Linear(base.in_features, rank, bias=False)
        self.lora_B = nn.Linear(rank, base.out_features, bias=False)

        nn.init.kaiming_uniform_(self.lora_A.weight, a=5 ** 0.5)
        nn.init.zeros_(self.lora_B.weight)

        for p in self.base.parameters():
            p.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) + self.lora_B(self.lora_A(self.dropout(x))) * self.scale


def _match_any(name: str, patterns: Optional[Iterable[str]], default: bool) -> bool:
    if not patterns:
        return default
    return any(fnmatch.fnmatch(name, p, flags=_GLOB_FLAGS) for p in patterns)


def apply_lora(
    module: nn.Module,
    rank: int,
    alpha: int,
    dropout: float = 0.0,
    target_patterns: Optional[Iterable[str]] = None,
    exclude_patterns: Optional[Iterable[str]] = None,
    freeze_base: bool = False,
    prefix: str = "",
) -> None:
    """
    Recursively replace Linear layers with LoRALinear if name matches patterns.
    """
    if freeze_base:
        for p in module.parameters():
            p.requires_grad = False

    for name, child in module.named_children():
        full_name = f"{prefix}.{name}" if prefix else name

        if isinstance(child, nn.Linear):
            if _match_any(full_name, target_patterns, True) and not _match_any(full_name, exclude_patterns, False):
                setattr(module, name, LoRALinear(child, rank=rank, alpha=alpha, dropout=dropout))
            continue

        apply_lora(
            child,
            rank=rank,
            alpha=alpha,
            dropout=dropout,
            target_patterns=target_patterns,
            exclude_patterns=exclude_patterns,
            prefix=full_name,
        )
