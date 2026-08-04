"""Single-model break-count + output runner (one process, one config).

Usage: python -m paper_eval._runner <model_key> <on|off>
Prints a JSON line: {"key","mode","graphs","breaks","out_hash","error"}

GraphMend is enabled by setting the program flags + scope BEFORE importing the
model's transformers module, so the meta-importer routes its .py through the
GraphMend pipeline. Must run with the REPO jaclang on PYTHONPATH.
"""
import sys, os, json, warnings, hashlib
warnings.filterwarnings("ignore")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import jaclang  # noqa: F401  installs the meta-importer hook
import torch

from paper_eval.registry import MODELS  # noqa: E402


def main(key: str, mode: str) -> None:
    spec = MODELS[key]
    if mode == "on":
        from jaclang.jac0core.runtime import JacRuntime as Jac
        prog = Jac.get_program()
        prog._graphmend_enabled = True
        prog._graphmend_scope = list(spec["scope"])

    torch.manual_seed(0)
    model, inputs = spec["build"]()
    model.eval()

    graphs = []

    def backend(gm, example_inputs):
        graphs.append(gm)
        return gm.forward

    torch._dynamo.reset()
    compiled = torch.compile(model, backend=backend, dynamic=False)
    with torch.no_grad():
        out = compiled(**inputs)

    # output fingerprint for correctness comparison (logits / last_hidden_state)
    t = None
    for attr in ("logits", "last_hidden_state"):
        if hasattr(out, attr):
            t = getattr(out, attr); break
    if t is None and isinstance(out, (tuple, list)):
        t = out[0]
    h = hashlib.sha256(t.detach().float().cpu().numpy().tobytes()).hexdigest()[:16] if t is not None else None
    # Report the input shape actually used. The paper states that the original
    # and transformed runs share a batch size and identical inputs; emitting the
    # shape per mode makes that checkable from the artifact rather than asserted,
    # and the off/on rows must agree.
    shape = None
    for v in inputs.values():
        if hasattr(v, "shape"):
            shape = list(v.shape)
            break
    print(json.dumps({"key": key, "mode": mode, "graphs": len(graphs),
                      "breaks": max(0, len(graphs) - 1), "out_hash": h,
                      "in_shape": shape, "error": None}))


if __name__ == "__main__":
    try:
        main(sys.argv[1], sys.argv[2])
    except Exception as e:
        import traceback; traceback.print_exc(file=sys.stderr)
        print(json.dumps({"key": sys.argv[1], "mode": sys.argv[2], "graphs": None,
                          "breaks": None, "out_hash": None, "error": f"{type(e).__name__}: {e}"}))
