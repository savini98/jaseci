"""GraphMend predicated trap-lowering (validation guard) tests.

Assertions run against the generated Python source (the lowering target)."""
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

test "torch.equal guard lowers to the tensor-equality assert helper" {
    src = _pysrc("val_guard.jac", True);
    assert (
        "__jac_tensor_eq_assert__(attention_mask, expected, 'unsupported attention mask')" in src
    );
    # helper import is emitted and the raising branch is gone
    assert "import __jac_tensor_eq_assert__" in src;
    assert "raise ValueError" not in src;
    assert "if not torch.equal" not in src;
}

test "tensor-bool guard lowers to torch._assert_async directly" {
    src = _pysrc("val_guard_all.jac", True);
    assert "torch._assert_async(mask.all(), 'mask must be all true')" in src;
    assert "raise ValueError" not in src;
    assert "if not mask.all()" not in src;
}

test "validation guard preserved when graphmend disabled" {
    src = _pysrc("val_guard.jac", False);
    assert "__jac_tensor_eq_assert__(" not in src;
    assert "if not torch.equal(attention_mask, expected):" in src;
    assert "raise ValueError('unsupported attention mask')" in src;
}
