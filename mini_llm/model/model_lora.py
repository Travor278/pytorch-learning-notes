from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass(frozen=True)
class LoRAConfig:
    rank: int = 8
    alpha: int = 16
    dropout: float = 0.05
    target_modules: tuple[str, ...] = ("q_proj", "v_proj")


class LoRALinear(nn.Module):
    def __init__(self, base_layer: nn.Linear, config: LoRAConfig):
        super().__init__()
        if config.rank <= 0:
            raise ValueError("LoRA rank must be positive.")
        self.base_layer = base_layer
        self.rank = config.rank
        self.alpha = config.alpha
        self.scaling = config.alpha / config.rank
        self.dropout = nn.Dropout(config.dropout)
        self.lora_A = nn.Linear(base_layer.in_features, config.rank, bias=False)
        self.lora_B = nn.Linear(config.rank, base_layer.out_features, bias=False)

        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)
        self.lora_A.to(device=base_layer.weight.device, dtype=base_layer.weight.dtype)
        self.lora_B.to(device=base_layer.weight.device, dtype=base_layer.weight.dtype)
        for param in self.base_layer.parameters():
            param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base_layer(x) + self.lora_B(self.lora_A(self.dropout(x))) * self.scaling

    def merge(self) -> nn.Linear:
        delta = self.lora_B.weight @ self.lora_A.weight
        self.base_layer.weight.data.add_(delta.to(self.base_layer.weight.dtype) * self.scaling)
        return self.base_layer


def _matches_target(module_name: str, target_modules: tuple[str, ...]) -> bool:
    short_name = module_name.rsplit(".", 1)[-1]
    return short_name in target_modules or module_name in target_modules


def apply_lora(
    model: nn.Module,
    *,
    rank: int = 8,
    alpha: int = 16,
    dropout: float = 0.05,
    target_modules: tuple[str, ...] = ("q_proj", "v_proj"),
) -> nn.Module:
    config = LoRAConfig(rank=rank, alpha=alpha, dropout=dropout, target_modules=target_modules)

    replacements: list[tuple[nn.Module, str, nn.Linear]] = []
    for module_name, module in model.named_modules():
        for child_name, child in module.named_children():
            full_name = f"{module_name}.{child_name}" if module_name else child_name
            if isinstance(child, nn.Linear) and _matches_target(full_name, config.target_modules):
                replacements.append((module, child_name, child))

    for parent, child_name, child in replacements:
        setattr(parent, child_name, LoRALinear(child, config))

    mark_only_lora_as_trainable(model)
    return model


def mark_only_lora_as_trainable(model: nn.Module) -> None:
    for name, param in model.named_parameters():
        param.requires_grad = "lora_A" in name or "lora_B" in name


def merge_lora_weights(model: nn.Module) -> nn.Module:
    for module in model.modules():
        for child_name, child in list(module.named_children()):
            if isinstance(child, LoRALinear):
                setattr(module, child_name, child.merge())
    return model


def lora_parameter_count(model: nn.Module) -> int:
    return sum(param.numel() for param in model.parameters() if param.requires_grad)
