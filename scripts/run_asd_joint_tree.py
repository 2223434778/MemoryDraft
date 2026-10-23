"""Second stage: regret-conditioned spine allocation with unchanged acceptance."""
from dataclasses import replace
import os
import time

from memorydraft.evaluation.asd_joint_tree import ARMS, STUDY, LegacyCheck, frozen_config
from scripts import run_asd_tree as common
from scripts.diagnose_transport_online import write


def tree_factory(engine,warm,out):
    shard = int(os.environ["CUDA_VISIBLE_DEVICES"])
    config, provenance = frozen_config(common.ROOT,shard)
    write(out/"cost-table.json",dict(**provenance,profile="first-stage FP32/RMSNorm A6000; read only"))
    return config


def execute(engine,record,arm,limit,tree_config,*,zero=False):
    if arm not in ARMS:
        raise ValueError("Unknown joint-tree arm")
    engine.synchronize()
    started = time.perf_counter()
    original = arm
    if arm in ("SpineTree-ASD","JointTree-ASD"):
        tree_config = replace(tree_config,spine_mode="full" if arm == "SpineTree-ASD" else "remaining")
        arm = "BudgetTree-ASD"
    row = common.execute(engine,record,arm,limit,tree_config,zero=zero)
    row.update(arm=original,method=original,latency_seconds=time.perf_counter()-started)
    return row


def main():
    common.main(study=STUDY,arms=ARMS,tree_factory=tree_factory,execution=execute,
                legacy_check=LegacyCheck(common.ROOT))


if __name__ == "__main__":main()
