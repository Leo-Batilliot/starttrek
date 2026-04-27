from __future__ import annotations

import torch.nn as nn


_ACTIVATIONS: dict[str, type[nn.Module]] = {
    "relu": nn.ReLU,
    "tanh": nn.Tanh,
    "gelu": nn.GELU,
}


def build_mlp(
    input_dim: int,
    output_dim: int,
    hidden_sizes: list[int],
    activation: str = "relu",
) -> nn.Sequential:
    if input_dim <= 0:
        raise ValueError(f"input_dim must be > 0, got {input_dim}.")
    if output_dim <= 0:
        raise ValueError(f"output_dim must be > 0, got {output_dim}.")
    if any(h <= 0 for h in hidden_sizes):
        raise ValueError(f"hidden_sizes must contain positive ints, got {hidden_sizes}.")
    if activation not in _ACTIVATIONS:
        raise ValueError(
            f"Unknown activation '{activation}'. Supported: {sorted(_ACTIVATIONS)}."
        )

    act_cls = _ACTIVATIONS[activation]

    layers: list[nn.Module] = []
    prev = input_dim
    for h in hidden_sizes:
        layers.append(nn.Linear(prev, h))
        layers.append(act_cls())
        prev = h
    layers.append(nn.Linear(prev, output_dim))

    net = nn.Sequential(*layers)
    _init_weights(net, activation)
    return net


def _init_weights(net: nn.Sequential, activation: str) -> None:
    use_kaiming = activation in ("relu", "gelu")
    nonlinearity = "relu" if use_kaiming else "tanh"

    for m in net.modules():
        if isinstance(m, nn.Linear):
            if use_kaiming:
                nn.init.kaiming_uniform_(m.weight, nonlinearity=nonlinearity)
            else:
                nn.init.xavier_uniform_(m.weight)
            nn.init.zeros_(m.bias)
