#!/usr/bin/env python
# ruff: noqa: T201  -- CLI script: console output is the point.
"""Measure GraphMend's wall-clock overhead.

Compiles each input file with and without GraphMend and reports:

  - per-pass wall-clock time of the four GraphMend passes (median over repeats),
    as recorded by the pass framework itself (``BaseTransform.time_taken``,
    collected via the opt-in ``JacProgram._pass_times`` hook in run_schedule);
  - total compile wall-clock time with GraphMend on vs. off, and the delta.

By default the module is treated as a compiled region (``--graphmend_scope``
semantics, ``_graphmend_scoped_compile``), which is how GraphMend processes
real HuggingFace modeling files end-to-end -- so the passes do their full
detection + transformation work rather than gating out on a missing
``torch.compile`` entry point. Pass ``--entry-only`` to disable that and
measure entry-file semantics instead.

Usage (from the repo root):

  PYTHONPATH=jac python jac/scripts/benchmark_graphmend.py [files...] \
      [--repeats N] [--entry-only]

With no files, benchmarks the transformers modeling files used in the paper's
multi-model evaluation (requires ``transformers`` installed).
"""

from __future__ import annotations

import argparse
import statistics
import time
from pathlib import Path

from jaclang.jac0core.compile_options import CompileOptions
from jaclang.jac0core.program import JacProgram

GM_PASSES = (
    "GraphBreakDetectPass",
    "TrapLoweringPass",
    "PredicateCtrlFlowPass",
    "DeferSideEffectPass",
)

DEFAULT_MODELS = (
    "pegasus",
    "bart",
    "biogpt",
    "blenderbot",
    "marian",
    "t5",
    "whisper",
)


def default_files() -> list[Path]:
    import transformers

    root = Path(transformers.__file__).parent / "models"
    return [root / m / f"modeling_{m}.py" for m in DEFAULT_MODELS]


def compile_once(
    path: Path, graphmend: bool, whole_module: bool
) -> tuple[float, dict[str, float]]:
    """One fresh compile; returns (total seconds, gm-pass-name -> seconds)."""
    prog = JacProgram()
    prog._pass_times = []
    if graphmend and whole_module:
        prog._graphmend_scoped_compile = True
    t0 = time.perf_counter()
    prog.compile(
        file_path=str(path),
        options=CompileOptions(graphmend=graphmend, type_check=False),
    )
    total = time.perf_counter() - t0
    gm = dict.fromkeys(GM_PASSES, 0.0)
    for name, secs in prog._pass_times:
        if name in gm:
            gm[name] += secs
    return total, gm


def bench(path: Path, repeats: int, whole_module: bool) -> None:
    loc = sum(1 for _ in path.open())
    # Warm-up: first compile in a process pays one-time import/parser costs.
    compile_once(path, True, whole_module)
    compile_once(path, False, whole_module)

    on_totals: list[float] = []
    off_totals: list[float] = []
    per_pass: dict[str, list[float]] = {n: [] for n in GM_PASSES}
    for _ in range(repeats):
        total, gm = compile_once(path, True, whole_module)
        on_totals.append(total)
        for name in GM_PASSES:
            per_pass[name].append(gm[name])
        total, _ = compile_once(path, False, whole_module)
        off_totals.append(total)

    med_on = statistics.median(on_totals)
    med_off = statistics.median(off_totals)
    med_pass = {n: statistics.median(per_pass[n]) for n in GM_PASSES}
    gm_total = sum(med_pass.values())

    print(f"\n{path.name}  ({loc} LOC, {repeats} repeats, median)")
    for name in GM_PASSES:
        print(f"  {name:<24} {med_pass[name] * 1000:9.2f} ms")
    print(
        f"  {'GraphMend total':<24} {gm_total * 1000:9.2f} ms"
        f"  ({100 * gm_total / med_on:.1f}% of compile)"
    )
    print(f"  {'compile (graphmend on)':<24} {med_on * 1000:9.2f} ms")
    print(f"  {'compile (graphmend off)':<24} {med_off * 1000:9.2f} ms")
    print(f"  {'on/off delta':<24} {(med_on - med_off) * 1000:9.2f} ms")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("files", nargs="*", type=Path)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument(
        "--entry-only",
        action="store_true",
        help="do not treat the module as a compiled region "
        "(entry-file semantics: needs a torch.compile entry point in-file)",
    )
    args = ap.parse_args()

    files = args.files or default_files()
    for f in files:
        if not f.exists():
            print(f"skip (not found): {f}")
            continue
        bench(f, args.repeats, whole_module=not args.entry_only)


if __name__ == "__main__":
    main()
