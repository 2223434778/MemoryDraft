"""Reuse paired compilation analysis, with RMSNorm identities and diagnostics."""
from scripts import write_asd_compile_report as common
from scripts.run_asd_rmsnorm_probe import PROFILES, expected_states


def audit(rows, preflight, diagnostics, previous):
    return common.audit(rows, preflight, diagnostics, previous, profiles=PROFILES,
                        replay_key="rmsnorm_replays", expected_states=expected_states())


def summarize(rows, preflight, diagnostics):
    return common.summarize(rows, preflight, diagnostics, profiles=PROFILES,
                            replay_key="rmsnorm_replays", seed="rmsnorm")


if __name__ == "__main__":
    common.main(study="RMSNORM", region="RMSNorm", profiles=PROFILES,
                replay_key="rmsnorm_replays", expected_states=expected_states(), seed="rmsnorm")
