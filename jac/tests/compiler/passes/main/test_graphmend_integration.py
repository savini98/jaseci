"""Torch-gated integration tests for GraphMend.

Confirm the *runtime* claim that the transformations defragment the FX graph:
compile each fixture with and without ``--graphmend`` and count how many
contiguous graphs TorchDynamo's backend actually receives. This is the paper's
metric -- a data-dependent branch / validation guard / side-effect call splits
the forward pass into multiple graphs; GraphMend collapses it back to one.

Graph count (via a counting backend) is used rather than
``explain().graph_break_count`` (which under-reports for small functions) or
``fullgraph=True`` (which is stricter than how models actually run -- it rejects
the harmless *trailing* break left by the side-effect epilogue flush even though
the expensive compute is a single contiguous graph).

Skipped automatically when PyTorch is not installed.
"""

import ast
import os

import pytest

torch = pytest.importorskip("torch")

from jaclang.jac0core.compile_options import CompileOptions  # noqa: E402
from jaclang.jac0core.program import JacProgram  # noqa: E402

pytestmark = pytest.mark.filterwarnings("ignore")

FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures", "graphmend")


def _compile_src(fixture: str, graphmend: bool) -> str:
    prog = JacProgram()
    mod = prog.compile(
        file_path=os.path.join(FIXTURES, fixture),
        options=CompileOptions(graphmend=graphmend, type_check=False),
    )
    py = mod.gen.py_ast
    py = py[0] if isinstance(py, list) else py
    return ast.unparse(py)


def _count_graphs(fixture: str, graphmend: bool, fn_name: str, *args: object) -> int:
    """Compile a fixture and count the contiguous graphs the backend receives.

    ``@torch.compile`` is neutralized to an identity decorator so the rewritten
    body is what we drive through a counting backend ourselves.
    """
    src = _compile_src(fixture, graphmend)
    ns: dict = {}
    orig_compile = torch.compile
    torch.compile = lambda *a, **k: a[0] if a else (lambda f: f)
    try:
        exec(compile(src, fixture, "exec"), ns)
    finally:
        torch.compile = orig_compile

    graphs: list = []

    def backend(gm: object, inputs: object) -> object:
        graphs.append(gm)
        return gm.forward  # type: ignore[attr-defined]

    torch._dynamo.reset()
    torch.compile(ns[fn_name], backend=backend)(*args)
    return len(graphs)


def _assert_defragmented(fixture: str, fn_name: str, *args: object) -> None:
    orig = _count_graphs(fixture, False, fn_name, *args)
    fixed = _count_graphs(fixture, True, fn_name, *args)
    assert orig >= 2, (
        f"{fixture}: expected a fragmented graph (>=2) without GraphMend, got {orig}"
    )
    assert fixed == 1, (
        f"{fixture}: expected one contiguous graph with GraphMend, got {fixed}"
    )


def test_data_dependent_return_defragmented():
    a = torch.tensor([1.0, 2.0, 3.0])
    b = torch.tensor([4.0, 5.0, 6.0])
    _assert_defragmented("where_return.jac", "with_breaks", a, b)


def test_data_dependent_assignment_defragmented():
    x = torch.tensor([1.0, 2.0, 3.0])
    y = torch.tensor([4.0, 5.0, 6.0])
    _assert_defragmented("where_assign.jac", "f", x, y)


def test_validation_guard_defragmented():
    x = torch.tensor([1.0, 2.0, 3.0])
    mask = torch.tensor([True, True, True])
    _assert_defragmented("val_guard_all.jac", "f", x, mask)


def test_print_side_effect_defragmented():
    x = torch.tensor([1.0, 2.0, 3.0])
    _assert_defragmented("side_effect.jac", "f", x)


def test_logger_side_effect_defragmented():
    x = torch.tensor([1.0, 2.0, 3.0])
    _assert_defragmented("side_effect_logger.jac", "f", x)
