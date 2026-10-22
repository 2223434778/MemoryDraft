"""Copy an already prepared, SHA-pinned local adapter; never download weights."""
import argparse
from pathlib import Path

from memorydraft.runtime.config import RuntimeConfig
from memorydraft.runtime.draft_adapter import DraftAdapterSpec

ROOT = Path(__file__).resolve().parents[1]


def prepare(source, config_path):
    config = RuntimeConfig.load(config_path)
    if not config.draft_adapter:
        raise ValueError("Choose an adapter-enabled runtime config")
    spec = DraftAdapterSpec(**config.draft_adapter)
    DraftAdapterSpec(str(source), spec.sha256).verify()
    destination = Path(spec.path)
    if destination.exists():
        spec.verify()
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("xb") as stream:
        stream.write(source.read_bytes())
    spec.verify()
    return destination


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=ROOT/"configs/runtime-dflash-asd-position.json")
    args = parser.parse_args()
    print(prepare(args.source, args.config))
