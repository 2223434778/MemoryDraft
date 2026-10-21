"""RMSNorm-only study, sharing the frozen MLP experiment's measurement boundary."""
from contextlib import contextmanager

from memorydraft.backends.regional_compile import RegionalRMSNorm
from scripts import run_asd_compile_probe as common

PROFILES = ("eager", "rmsnorm-inductor")
# Capture one representative of every named role, rather than every layer.
ROLES = ("input_layernorm", "post_attention_layernorm", "q_norm", "k_norm", "norm")


def representatives(compiler):
    selected = []
    for owner in ("target", "draft"):
        for role in (*ROLES, *(("hidden_norm",) if owner == "draft" else ())):
            matches = [(n, m) for n, m in compiler.regions if n.startswith(owner+".") and n.split(".")[-1] == role]
            if not matches:
                raise ValueError("Missing audited RMSNorm role: "+owner+"."+role)
            selected.append((*matches[0], role))
    return selected


@contextmanager
def capture_inputs(compiler):
    states, handles = [], []
    try:
        for name, module, role in representatives(compiler):
            calls = [0]

            def hook(m, args, *, name=name, calls=calls):
                index = calls[0]
                calls[0] += 1
                # Target prefill + first verify; Draft first invocation.
                if index < (2 if name.startswith("target.") else 1):
                    x = args[0]
                    states.append(dict(region=name, call=index, module=m, input=x.detach().clone(),
                                       original_stride=list(x.stride())))

            handles.append(module.register_forward_pre_hook(hook))
        yield states
    finally:
        for handle in handles:
            handle.remove()


def expected_states():
    return sorted([(owner, role, call) for owner in ("target", "draft")
                   for role in (*ROLES, *(("hidden_norm",) if owner == "draft" else ()))
                   for call in range(2 if owner == "target" else 1)])


if __name__ == "__main__":
    common.main(compiler_type=RegionalRMSNorm, profiles=PROFILES, capture=capture_inputs,
                replay_key="rmsnorm_replays", state_key="rmsnorm_states")
