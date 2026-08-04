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


def _transform_remote_module(mod_name: str) -> None:
    """Apply GraphMend to an already-imported Hub remote-code module, in place.

    `--graphmend-scope` cannot reach Hub remote code: transformers loads it with
    `spec_from_file_location` + `exec_module`, which bypasses `sys.meta_path`
    entirely, so jaclang's importer never sees the module. The transformation
    itself applies fine -- only the delivery path differs -- so compile the
    module's source through the same pipeline and re-exec the result over it.
    """
    import ast, types
    from jaclang.jac0core.program import JacProgram
    from jaclang.jac0core.compile_options import CompileOptions

    target = next((n for n in sys.modules if n.endswith(mod_name)), None)
    if target is None:
        raise RuntimeError(f"remote module {mod_name!r} not imported")
    orig = sys.modules[target]
    src_file = orig.__file__
    prog = JacProgram()
    prog._graphmend_enabled = True
    prog._graphmend_scoped_compile = True
    compiled = prog.compile(
        file_path=src_file,
        options=CompileOptions(graphmend=True, type_check=False),
    )
    pa = compiled.gen.py_ast
    src = ast.unparse(pa[0] if isinstance(pa, list) else pa)
    if "__jac_tensor_eq_assert__" not in src:
        raise RuntimeError(f"{mod_name}: GraphMend produced no [Trap] lowering")
    ns = types.ModuleType(target)
    ns.__file__ = src_file
    # Carry the hash transformers stamps on a loaded remote module. Without it
    # the next get_class_in_module sees a mismatch and re-execs the ORIGINAL
    # file over this namespace, silently undoing the transformation.
    ns.__transformers_module_hash__ = getattr(orig, "__transformers_module_hash__", "")
    sys.modules[target] = ns
    exec(compile(src, src_file, "exec"), ns.__dict__)  # noqa: S102


def main(key: str, mode: str) -> None:
    spec = MODELS[key]
    remote = spec.get("remote_code")
    if mode == "on":
        from jaclang.jac0core.runtime import JacRuntime as Jac
        prog = Jac.get_program()
        prog._graphmend_enabled = True
        prog._graphmend_scope = list(spec["scope"])

    torch.manual_seed(0)
    if mode == "on" and remote:
        # Build once so the remote source is resolved and imported, transform it,
        # then rebuild so the model is constructed from the transformed classes.
        spec["build"]()
        _transform_remote_module(remote)
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
    print(json.dumps({"key": key, "mode": mode, "graphs": len(graphs),
                      "breaks": max(0, len(graphs) - 1), "out_hash": h, "error": None}))


if __name__ == "__main__":
    try:
        main(sys.argv[1], sys.argv[2])
    except Exception as e:
        import traceback; traceback.print_exc(file=sys.stderr)
        print(json.dumps({"key": sys.argv[1], "mode": sys.argv[2], "graphs": None,
                          "breaks": None, "out_hash": None, "error": f"{type(e).__name__}: {e}"}))
