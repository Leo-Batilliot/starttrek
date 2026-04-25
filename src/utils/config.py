from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


def _require(d: Any, key: str, yaml_path: str) -> Any:
    if not isinstance(d, dict):
        raise ValueError(
            f"Expected a mapping at YAML path '{yaml_path}', got {type(d).__name__}."
        )
    if key not in d:
        raise ValueError(
            f"Missing required key '{yaml_path}.{key}' in YAML config."
        )
    return d[key]


@dataclass(frozen=True)
class WrapperConfig:
    name: str
    kwargs: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: Any, yaml_path: str) -> "WrapperConfig":
        if not isinstance(d, dict):
            raise ValueError(
                f"Expected a mapping at YAML path '{yaml_path}', got {type(d).__name__}."
            )
        return cls(
            name=str(_require(d, "name", yaml_path)),
            kwargs=dict(d.get("kwargs") or {}),
        )

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "kwargs": dict(self.kwargs)}


@dataclass(frozen=True)
class EnvConfig:
    id: str
    kwargs: dict[str, Any] = field(default_factory=dict)
    wrappers: tuple[WrapperConfig, ...] = field(default_factory=tuple)

    @classmethod
    def from_dict(cls, d: Any, yaml_path: str) -> "EnvConfig":
        if not isinstance(d, dict):
            raise ValueError(
                f"Expected a mapping at YAML path '{yaml_path}', got {type(d).__name__}."
            )
        raw_wrappers = d.get("wrappers") or []
        if not isinstance(raw_wrappers, list):
            raise ValueError(
                f"Expected a list at YAML path '{yaml_path}.wrappers', "
                f"got {type(raw_wrappers).__name__}."
            )
        wrappers = tuple(
            WrapperConfig.from_dict(w, f"{yaml_path}.wrappers[{i}]")
            for i, w in enumerate(raw_wrappers)
        )
        return cls(
            id=str(_require(d, "id", yaml_path)),
            kwargs=dict(d.get("kwargs") or {}),
            wrappers=wrappers,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "kwargs": dict(self.kwargs),
            "wrappers": [w.to_dict() for w in self.wrappers],
        }


@dataclass(frozen=True)
class NetworkConfig:
    hidden_sizes: tuple[int, ...]
    activation: str

    @classmethod
    def from_dict(cls, d: Any, yaml_path: str) -> "NetworkConfig":
        if not isinstance(d, dict):
            raise ValueError(
                f"Expected a mapping at YAML path '{yaml_path}', got {type(d).__name__}."
            )
        sizes = _require(d, "hidden_sizes", yaml_path)
        if not isinstance(sizes, list) or not all(isinstance(x, int) for x in sizes):
            raise ValueError(
                f"'{yaml_path}.hidden_sizes' must be a list of int, got {sizes!r}."
            )
        return cls(
            hidden_sizes=tuple(sizes),
            activation=str(_require(d, "activation", yaml_path)),
        )

    def to_dict(self) -> dict[str, Any]:
        return {"hidden_sizes": list(self.hidden_sizes), "activation": self.activation}


@dataclass(frozen=True)
class EpsilonConfig:
    start: float
    end: float
    decay_steps: int

    @classmethod
    def from_dict(cls, d: Any, yaml_path: str) -> "EpsilonConfig":
        if not isinstance(d, dict):
            raise ValueError(
                f"Expected a mapping at YAML path '{yaml_path}', got {type(d).__name__}."
            )
        return cls(
            start=float(_require(d, "start", yaml_path)),
            end=float(_require(d, "end", yaml_path)),
            decay_steps=int(_require(d, "decay_steps", yaml_path)),
        )

    def to_dict(self) -> dict[str, Any]:
        return {"start": self.start, "end": self.end, "decay_steps": self.decay_steps}


@dataclass(frozen=True)
class TargetUpdateConfig:
    kind: str
    every_steps: int = 0
    tau: float = 0.0

    @classmethod
    def from_dict(cls, d: Any, yaml_path: str) -> "TargetUpdateConfig":
        if not isinstance(d, dict):
            raise ValueError(
                f"Expected a mapping at YAML path '{yaml_path}', got {type(d).__name__}."
            )
        kind = str(_require(d, "kind", yaml_path))
        if kind not in ("hard", "soft"):
            raise ValueError(
                f"'{yaml_path}.kind' must be 'hard' or 'soft', got '{kind}'."
            )
        return cls(
            kind=kind,
            every_steps=int(d.get("every_steps", 0)),
            tau=float(d.get("tau", 0.0)),
        )

    def to_dict(self) -> dict[str, Any]:
        return {"kind": self.kind, "every_steps": self.every_steps, "tau": self.tau}


@dataclass(frozen=True)
class AgentConfig:
    name: str
    network: NetworkConfig
    gamma: float
    lr: float
    batch_size: int
    buffer_size: int
    min_buffer_size: int
    target_update: TargetUpdateConfig
    epsilon: EpsilonConfig
    grad_clip: float

    @classmethod
    def from_dict(cls, d: Any, yaml_path: str) -> "AgentConfig":
        if not isinstance(d, dict):
            raise ValueError(f"Expected a mapping at YAML path '{yaml_path}', got {type(d).__name__}.")
        return cls(
            name=str(_require(d, "name", yaml_path)),
            network=NetworkConfig.from_dict(
                _require(d, "network", yaml_path), f"{yaml_path}.network"
            ),
            gamma=float(_require(d, "gamma", yaml_path)),
            lr=float(_require(d, "lr", yaml_path)),
            batch_size=int(_require(d, "batch_size", yaml_path)),
            buffer_size=int(_require(d, "buffer_size", yaml_path)),
            min_buffer_size=int(_require(d, "min_buffer_size", yaml_path)),
            target_update=TargetUpdateConfig.from_dict(
                _require(d, "target_update", yaml_path),
                f"{yaml_path}.target_update",
            ),
            epsilon=EpsilonConfig.from_dict(
                _require(d, "epsilon", yaml_path), f"{yaml_path}.epsilon"
            ),
            grad_clip=float(_require(d, "grad_clip", yaml_path)),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "network": self.network.to_dict(),
            "gamma": self.gamma,
            "lr": self.lr,
            "batch_size": self.batch_size,
            "buffer_size": self.buffer_size,
            "min_buffer_size": self.min_buffer_size,
            "target_update": self.target_update.to_dict(),
            "epsilon": self.epsilon.to_dict(),
            "grad_clip": self.grad_clip,
        }


@dataclass(frozen=True)
class TrainingConfig:
    total_steps: int
    log_every: int
    eval_every: int
    eval_episodes: int
    save_every: int

    @classmethod
    def from_dict(cls, d: Any, yaml_path: str) -> "TrainingConfig":
        if not isinstance(d, dict):
            raise ValueError(f"Expected a mapping at YAML path '{yaml_path}', got {type(d).__name__}.")
        return cls(
            total_steps=int(_require(d, "total_steps", yaml_path)),
            log_every=int(_require(d, "log_every", yaml_path)),
            eval_every=int(_require(d, "eval_every", yaml_path)),
            eval_episodes=int(_require(d, "eval_episodes", yaml_path)),
            save_every=int(_require(d, "save_every", yaml_path)),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_steps": self.total_steps,
            "log_every": self.log_every,
            "eval_every": self.eval_every,
            "eval_episodes": self.eval_episodes,
            "save_every": self.save_every,
        }


@dataclass(frozen=True)
class EvalConfig:
    episodes: int
    greedy: bool
    record_video: bool
    video_every: int

    @classmethod
    def from_dict(cls, d: Any, yaml_path: str) -> "EvalConfig":
        if not isinstance(d, dict):
            raise ValueError(
                f"Expected a mapping at YAML path '{yaml_path}', got {type(d).__name__}."
            )
        return cls(
            episodes=int(_require(d, "episodes", yaml_path)),
            greedy=bool(_require(d, "greedy", yaml_path)),
            record_video=bool(_require(d, "record_video", yaml_path)),
            video_every=int(_require(d, "video_every", yaml_path)),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "episodes": self.episodes,
            "greedy": self.greedy,
            "record_video": self.record_video,
            "video_every": self.video_every,
        }


@dataclass(frozen=True)
class RunConfig:
    run_name: str
    seed: int
    env: EnvConfig
    agent: AgentConfig
    training: TrainingConfig
    eval: EvalConfig

    @classmethod
    def from_dict(cls, d: Any, yaml_path: str = "<root>") -> "RunConfig":
        if not isinstance(d, dict):
            raise ValueError(f"YAML root must be a mapping, got {type(d).__name__}.")
        return cls(
            run_name=str(_require(d, "run_name", yaml_path)),
            seed=int(_require(d, "seed", yaml_path)),
            env=EnvConfig.from_dict(_require(d, "env", yaml_path), f"{yaml_path}.env"),
            agent=AgentConfig.from_dict(_require(d, "agent", yaml_path), f"{yaml_path}.agent"),
            training=TrainingConfig.from_dict(_require(d, "training", yaml_path), f"{yaml_path}.training"),
            eval=EvalConfig.from_dict(_require(d, "eval", yaml_path), f"{yaml_path}.eval"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_name": self.run_name,
            "seed": self.seed,
            "env": self.env.to_dict(),
            "agent": self.agent.to_dict(),
            "training": self.training.to_dict(),
            "eval": self.eval.to_dict(),
        }


def load_config(path: str | Path) -> RunConfig:
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    if raw is None:
        raise ValueError(f"YAML file '{p}' is empty.")
    return RunConfig.from_dict(raw, yaml_path=str(p))


def save_config(cfg: RunConfig, path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg.to_dict(), f, sort_keys=False, default_flow_style=False)

if __name__ == "__main__":
    conf = load_config(Path("configs/baseline/dqn.yaml"))

    print(conf.training.eval_every)