"""Checked deployment assets for the existing cost-controlled DFlash tree."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
from importlib import resources
import json
from pathlib import Path

import torch
import transformers

from memorydraft.backends.budget_tree import BudgetTreeConfig, TreeCosts


def cost_environment(engine):
    """Use relocatable model identities; never equate different GPU profiles."""
    config = engine.config
    draft = json.loads(resources.files("memorydraft").joinpath("assets/dflash-manifest.json").read_text())
    return dict(target_revision=config.target_revision, draft_revision=draft["revision"],
                draft_adapter_sha256=(engine._adapter_info or {}).get("sha256"),
                torch=str(torch.__version__), transformers=transformers.__version__,
                gpu=torch.cuda.get_device_name(engine.target.device) if engine.target.device.type == "cuda" else "cpu",
                dtype=str(engine.target.dtype), attention=config.attention,
                execution_profile=config.execution_profile, threads=config.threads,
                draft_block_size=engine.draft.block_size)


@dataclass(frozen=True)
class CostTreeSpec:
    path: str
    sha256: str
    min_nodes: int = 8
    max_nodes: int = 32
    top_k: int = 4

    def __post_init__(self):
        if (not isinstance(self.path, str) or not self.path or not isinstance(self.sha256, str)
                or len(self.sha256) != 64 or any(c not in "0123456789abcdef" for c in self.sha256)):
            raise ValueError("Cost tree requires a local path and lowercase SHA-256")
        if (any(type(v) is not int for v in (self.min_nodes, self.max_nodes, self.top_k))
                or not 2 <= self.min_nodes <= self.max_nodes <= 64 or not 1 <= self.top_k <= 8):
            raise ValueError("Invalid cost tree bounds")

    def read(self):
        path = Path(self.path)
        if path.stat().st_size > 1024 * 1024:
            raise ValueError("Cost tree asset exceeds 1 MiB")
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != self.sha256:
            raise ValueError("Cost tree asset checksum mismatch")
        data = json.loads(raw)
        if (not isinstance(data, dict) or type(data.get("schema")) is not int or data.get("schema") != 1
                or not {"environment", "costs"} <= data.keys()
                or data.keys() - {"schema", "environment", "costs", "measurement"}
                or not isinstance(data["environment"], dict)):
            raise ValueError("Invalid cost tree asset schema")
        value = data["costs"]
        if not isinstance(value, dict) or set(value) != set(TreeCosts.__dataclass_fields__):
            raise ValueError("Invalid cost tree latency fields")
        try:
            if (any(type(v) is not int for v in (*value["contexts"], *value["nodes"]))
                    or any(type(v) not in (int, float) for row in value["seconds"] for v in row)
                    or any(type(value[k]) not in (int, float) for k in ("draft_seconds", "auxiliary_seconds"))):
                raise ValueError("Cost tree measurements must be numeric and dimensions integral")
            costs = TreeCosts(tuple(value["contexts"]), tuple(value["nodes"]),
                              tuple(tuple(row) for row in value["seconds"]),
                              value["draft_seconds"], value["auxiliary_seconds"])
        except (TypeError, KeyError) as error:
            raise ValueError("Malformed cost tree latency table") from error
        if self.min_nodes < costs.nodes[0] or self.max_nodes > costs.nodes[-1]:
            raise ValueError("Cost tree node bounds exceed the measured table")
        return BudgetTreeConfig(costs, self.min_nodes, self.max_nodes, self.top_k), data["environment"]

    def load(self, engine):
        config, environment = self.read()
        if not engine.config.target_revision:
            raise ValueError("Cost tree requires a pinned Target revision")
        expected = cost_environment(engine)
        if environment != expected:
            differing = sorted(k for k in environment.keys() | expected.keys() if environment.get(k) != expected.get(k))
            raise ValueError("Cost tree environment differs; recalibrate: " + ", ".join(differing))
        return config

    def describe(self):
        return {**asdict(self), "proposal": "cost_tree", "spine_mode": "none"}
