"""GraphMend deferred side-effect transformation tests.

Assertions run against the generated Python source (the lowering target)."""
import ast;
import os;
import from pathlib { Path }
import jaclang;
import from jaclang.jac0core.jaclib { __jac_flush_se_buffer__ }
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

test "print is deferred to a buffer and flushed after the graph" {
    src = _pysrc("side_effect.jac", True);
    assert "_gm_se_buffer = []" in src;
    assert "_gm_se_buffer.append((print, (a,), {}))" in src;
    assert "__jac_flush_se_buffer__(_gm_se_buffer)" in src;
    # flush helper import is emitted
    assert "import __jac_flush_se_buffer__" in src;
    # value captured before reassignment (append precedes `a = a + 1`)
    assert src.index("_gm_se_buffer.append") < src.index("a = a + 1");
}

test "dotted logger call keeps its keyword argument in the buffered entry" {
    src = _pysrc("logger_kwargs.jac", True);
    assert "_gm_se_buffer.append((logger.warning, ('deprecated',), {'stacklevel': 2}))" in src;
}

test "buffer is flushed before every return" {
    src = _pysrc("side_effect_early_return.jac", True);
    # one flush before the early return, one before the tail return
    assert src.count("__jac_flush_se_buffer__(_gm_se_buffer)") == 2;
}

test "side effects untouched when graphmend disabled" {
    src = _pysrc("side_effect.jac", False);
    assert "_gm_se_buffer" not in src;
    assert "print(a)" in src;
}

test "flush helper replays calls in order and clears the buffer" {
    out = [];
    buf = [(out.append, (1, ), {}), (out.append, (2, ), {})];
    __jac_flush_se_buffer__(buf);
    assert out == [1, 2];
    assert buf == [];
}
