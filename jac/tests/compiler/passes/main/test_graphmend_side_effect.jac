"""GraphMend deferred side-effect transformation tests.

print/logger calls are rewritten into appends on a PROCESS-GLOBAL buffer
(`__jac_se_emit__`), and the buffer is flushed after all compute (the return
value is hoisted into a temp so the flush is a trailing eager epilogue). This
keeps the forward pass a single contiguous FX graph while preserving the calls.

The buffer is global rather than function-local so it outlives a frame that
exits by raising: the `@__jac_se_guard__` boundary decorator, which sits ABOVE
`@torch.compile` and is therefore eager, drains it on that path. Assertions run
against the generated Python source.
"""
import ast;
import os;
import from pathlib { Path }
import jaclang;
import from jaclang.jac0core.program { JacProgram }
import from jaclang.jac0core.compile_options { CompileOptions }

glob FIXTURES = os.path.join(
         str(Path(jaclang.__file__).parent.parent),
         "tests",
         "compiler",
         "passes",
         "main",
         "fixtures",
         "graphmend"
     );

"""Compile a fixture and return its generated Python source."""
def _pysrc(name: str, gm: bool) -> str {
    prog = JacProgram();
    mod = prog.compile(
        file_path=os.path.join(FIXTURES, name),
        options=CompileOptions(graphmend=gm, type_check=False)
    );
    pa = mod.gen.py_ast;
    mod_ast = pa[0] if isinstance(pa, list) else pa;
    return ast.unparse(mod_ast);
}

test "print is buffered and flushed via the runtime helper" {
    src = _pysrc("side_effect.jac", True);
    # The mutable positional arg is snapshotted: hoisted to a temp and cloned (if
    # a tensor) at the call site so a later in-place mutation cannot corrupt the
    # deferred value (paper Table 1, clone(args)).
    assert "__gm_se_arg_0 = x" in src;
    assert (
        "__jac_se_emit__(print, "
        "(__gm_se_arg_0.clone() if hasattr(__gm_se_arg_0, 'clone') "
        "else __gm_se_arg_0,), {})"
    )
        in src;
    assert "__jac_se_flush__()" in src;
    assert "__jac_se_emit__" in src;
}

test "a compiled function gets the eager boundary guard" {
    src = _pysrc("side_effect.jac", True);
    # The guard must sit ABOVE @torch.compile: prepended, it wraps the compiled
    # callable from outside, so its try/finally is never traced. Placed inside
    # the compiled region an untraceable finally makes Dynamo abandon the frame.
    assert "@__jac_se_guard__" in src;
    assert src.index("@__jac_se_guard__") < src.index("@torch.compile");
}

test "return value is hoisted so the flush follows all compute" {
    src = _pysrc("side_effect.jac", True);
    # compute assigned to a temp, then flush, then return the temp
    assert "__gm_ret_0 = torch.sin(x) * 2.0" in src;
    assert src.index("__gm_ret_0 = ") < src.index("__jac_se_flush__()");
    assert src.index("__jac_se_flush__()") < src.index("return __gm_ret_0");
}

test "logger call is buffered the same way as print" {
    src = _pysrc("side_effect_logger.jac", True);
    assert "__jac_se_emit__(log.warning, ('deprecated path',), {})" in src;
    assert "__jac_se_flush__()" in src;
}

test "no buffering when graphmend disabled" {
    src = _pysrc("side_effect.jac", False);
    assert "__jac_se_emit__" not in src;
    assert "__jac_se_guard__" not in src;
    assert "print(x)" in src;
}
