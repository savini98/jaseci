"""GraphMend predicated control-flow (torch.where) transformation tests.

Assertions run against the generated Python source (the lowering target), which
is the reliable observable: the Jac unparser omits synthetic punctuation and the
`_sub_node_tab` cache is not refreshed after the rewrite.
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

test "return branches become torch.where" {
    src = _pysrc("where_return.jac", True);
    assert "torch.where(" in src;
    assert "return torch.where(" in src;
    assert "if " not in src;
}

test "assignment branches become torch.where" {
    src = _pysrc("where_assign.jac", True);
    assert "z = torch.where(" in src;
    assert "if " not in src;
}

test "common call branches merge differing arg via torch.where" {
    src = _pysrc("where_common_call.jac", True);
    assert "torch.where(" in src;
    # the differing arg is merged inside the single preserved call
    assert "register_buffer('inv_freq', torch.where(" in src;
    assert "persistent=False" in src;
    assert "if " not in src;
}

test "common call merges multiple differing args, preserves kwargs" {
    src = _pysrc("where_multi_arg.jac", True);
    # both differing positional args merged independently; kwarg kept
    assert "combine(torch.where(" in src;
    assert src.count("torch.where(") == 2;
    assert "mode='x'" in src;
    assert "if " not in src;
}

test "item() dynamic signal in return branch becomes torch.where" {
    src = _pysrc("where_item_return.jac", True);
    assert "x.item() > 5" in src;
    assert "return torch.where(" in src;
    assert "if " not in src;
}

test "different callees in branches are left untouched" {
    src = _pysrc("different_callees.jac", True);
    assert "torch.where(" not in src;
    assert "if x.sum() > 0:" in src;
    assert "foo(p)" in src;
    assert "bar(q)" in src;
}

test "multi-statement branches with shared call hoist and merge (Fig. 4)" {
    src = _pysrc("where_multi_stmt.jac", True);
    # data-dependent branch becomes a predicate assignment...
    assert "__gm_cond_0 = seq_len > omp" in src;
    assert "if seq_len > omp:" not in src;
    # ...with setups hoisted and the shared call emitted once via torch.where
    assert "register_buffer('inv_freq', torch.where(" in src;
    assert "self_obj.original_inv_freq = self_obj.original_inv_freq.to(device)" in src;
    # static hasattr guards are preserved (not rewritten)
    assert "if hasattr(self_obj.config" in src;
    assert "if not hasattr(self_obj, 'long_inv_freq')" in src;
}

test "control flow preserved when graphmend disabled" {
    src = _pysrc("where_return.jac", False);
    assert "torch.where(" not in src;
    assert "if " in src;
}
