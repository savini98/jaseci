"""GraphMend graph-break detection pass tests."""
import os;
import from pathlib { Path }
import jaclang;
import jaclang.jac0core.unitree as uni;
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

"""Compile a fixture with GraphMend optionally enabled."""
def _compile(name: str, gm: bool) -> uni.Module {
    prog = JacProgram();
    return prog.compile(
        file_path=os.path.join(FIXTURES, name),
        options=CompileOptions(graphmend=gm, type_check=False)
    );
}

"""Collect the GraphMend break-kind tag of every IfStmt in a module."""
def _if_kinds(mod: uni.Module) -> list {
    return [n?._gm_break_kind for n in mod.get_all_sub_nodes(uni.IfStmt)];
}

"""Collect non-empty GraphMend break-kind tags of FuncCalls in a module."""
def _call_kinds(mod: uni.Module) -> list {
    out = [];
    for n in mod.get_all_sub_nodes(uni.FuncCall) {
        k = n?._gm_break_kind;
        if k {
            out.append(k);
        }
    }
    return out;
}

test "detects data-dependent control flow" {
    mod = _compile("where_return.jac", True);
    assert "dyn_ctrl_fl" in _if_kinds(mod);
}

test "detects validation guard" {
    mod = _compile("val_guard.jac", True);
    assert "val_guard" in _if_kinds(mod);
}

test "detects print side effect" {
    mod = _compile("side_effect.jac", True);
    assert "side_effect" in _call_kinds(mod);
}

test "no detection when graphmend disabled" {
    mod = _compile("where_return.jac", False);
    assert _if_kinds(mod) == [None];
}

test "static hasattr control flow is not tagged" {
    mod = _compile("hasattr_static.jac", True);
    assert "dyn_ctrl_fl" not in _if_kinds(mod);
    assert "val_guard" not in _if_kinds(mod);
}
