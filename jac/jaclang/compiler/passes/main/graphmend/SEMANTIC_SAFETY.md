# GraphMend -- Transformation Legality & Semantic Safety

This document specifies *why* each GraphMend transformation is semantics-preserving:
the safety conditions it checks, as first-principles invariants, decision
flowcharts, and formal algorithms. It complements the paper's **Algorithm 1
(Fixable Graph-Break Detection)** with the missing **legality** algorithms that
decide whether a detected break can be *safely* rewritten.

It also doubles as rebuttal material: it reframes the three transformations from
"simple source rewrites" into a family of legality analyses resting on a stated
soundness obligation.

---

## 1. The unifying problem

All three transformations are instances of **one** operation: taking a
computation that originally executes *conditionally* (on some control path, or as
a runtime interaction) and re-expressing it as *unconditional dataflow* that
TorchDynamo can capture as a single FX graph. In classical compiler terms this is
**if-conversion** -- converting control dependence into data dependence -- plus
**effect reordering**.

The moment you do that you are **speculating**: executing work (or changing
*when* an effect happens) that the original program might not have. So every
analysis pass answers one question:

> Under what conditions is speculatively executing this region -- and
> muxing/deferring its results -- *observationally equivalent* to the original
> conditional execution?

Everything the passes check is a necessary condition for that equivalence.

---

## 2. The four safety invariants

| # | Invariant | Establishes |
|---|-----------|-------------|
| **INV1** | **Predicate well-formedness** -- the branch is a clean two-way join whose reachability is fully determined by a single condition that is materializable as a *value* and *pure* to evaluate. | A single `where`/`cond` mux faithfully stands in for the control join. |
| **INV2** | **Path totality under speculation** -- executing the *not-taken* path introduces no new fault, divergence, or exception that the original control flow excluded. | Speculation does not expose behavior the guard previously screened off. (This is why a `raise`/assert in a branch is unsafe to naively hoist -- you must re-predicate it or decline.) |
| **INV3** | **Effect neutrality / recoverability** -- every observable effect in the region is pure, reconcilable to a single predicated effect, or deferrable-and-replayable in original order under its original predicate; and the effecting call's binding is stable. | Observable behavior (I/O, logging, mutation) is preserved despite reordering. |
| **INV4** | **Join compatibility (SSA φ)** -- both paths define the same live outputs with a mux-compatible value domain (tensor, matching structure; not `None`/non-tensor). | The merge is expressible as a value-level select; `where`/`cond` is the φ made concrete. |

**Meta-invariant -- conservative soundness.** These properties are undecidable in
general, so the analysis is a **must-analysis**: a transformation fires only when
safety is *established*, and absence of proof is treated as unsafe. Coverage is
sacrificed for correctness. This is what lets us claim semantics-preservation as a
guarantee rather than a hope, and why "we leave some breaks unfixed" is a
soundness property, not a weakness.

### Mapping to classical theory

- Control-dependence → data-dependence conversion (Allen-Kennedy if-conversion).
- INV2 = classical **speculation legality**: control-speculating an op is legal
  only if it is *safe* (non-faulting, effect-free); otherwise **predication** or
  **recovery code** is required. GraphMend's re-gating of a hoisted assert is
  predicated recovery.
- INV4 = **SSA construction**; the mux is a materialized φ-function.
- INV3 = **effect analysis + reordering legality**, plus **name-resolution /
  points-to on callables** (binding stability).

The novel part: prior if-conversion theory assumes a static, typed,
side-effect-disciplined IR. GraphMend establishes these invariants over
**dynamically-typed Python with arbitrary rebinding and unrestricted side
effects** -- which is why it needs the unified `⟨AST, CFG, SymTab⟩` IR. INV1 needs
the CFG, INV2/INV4 need type/value reasoning, INV3 needs the symbol table, and
they must hold *simultaneously*. No single view suffices -- the argument that this
cannot be done at the bytecode level.

---

## 3. Analysis primitives

Everything operates over the unified IR `U = ⟨AST, CFG, ST⟩`. Each primitive is a
**conservative must-predicate** -- `true` only when provable over `U`; unprovable ⇒
`false`. `⊥` denotes **decline** (return the region unchanged, break left intact).

```
DataDep(e)        ≡ e uses a dynamic torch attr, or a use-def trace in ST∪CFG
                    reaches a tensor input / dynamic op            (Alg. 1)
Materializable(e) ≡ ⟦e⟧ is a first-class value (tensor/scalar), not only a
                    control decision
Pure(e)           ≡ evaluating e performs no observable effect
LiveOut(B)        ≡ { v : v defined in B, live at the post-join point }   (CFG)
TensorVal(e)      ≡ value-domain(e) is mux-compatible (¬None, ¬non-tensor)
CtrlFree(S)       ≡ ∄ s ∈ S* : s ∈ {return, raise, break, continue}
BindingStable(c)  ≡ callee(c) resolves in ST to the intended builtin/lib symbol
                    (not user-rebound)
Independent(s,R)  ≡ inputs(s) ∩ writes(R) = ∅ ∧ writes(s) ∩ inputs(R) = ∅
HasCall(e)        ≡ e contains a FuncCall
Writes(s, X.k)    ≡ s assigns attribute k of X (incl. as a tuple-target element)
Confined(X.k, R)  ≡ no reference to attribute `k` -- an `_.k` access, or a string
                    literal `"k"` that hasattr/getattr/setattr could consume --
                    occurs in the module outside region R  (module-scope must-check)
TensorBool(e)     ≡ e is a call returning a tensor bool (.all/.any/.allclose)
```

---

## 4. Per-transformation decision flowcharts

`[INVn]` tags map each gate to §2. Any failed gate exits to **⛔ DECLINE**.
Legend: `[≡]` = value-equivalence of a lowered check.

### 4.1 Predicated Data-Dependent Control Flow (`if/else → torch.where / torch.cond`)

```
        ┌────────────────────────────────────────────┐
        │ IfStmt tagged dyn_ctrl_fl                    │
        └─────────────────────┬──────────────────────-┘
            ┌─────────────────▼─────────────────┐  No
            │ Two-way join? has else, no elif    ├──────► ⛔ DECLINE
            │ chain                       [INV1] │
            └─────────────────┬─────────────────┘
                              │ Yes
            ┌─────────────────▼─────────────────┐  No
            │ Predicate materializable & pure    ├──────► ⛔ DECLINE
            │                             [INV1] │
            └─────────────────┬─────────────────┘
                              │ Yes
            ┌─────────────────▼─────────────────┐  No
            │ Same live outputs both paths?      ├──────► ⛔ DECLINE
            │ same LHS / return / callee  [INV4] │
            └─────────────────┬─────────────────┘
                              │ Yes
            ┌─────────────────▼─────────────────┐  Yes (None/
            │ Any branch value None/non-tensor?  ├─nontensor)► ⛔ DECLINE
            │                             [INV4] │
            └─────────────────┬─────────────────┘
                              │ No
            ┌─────────────────▼─────────────────┐  No (raise/
            │ Hoisted setups control-free?[INV2] ├─return/    ► ⛔ DECLINE
            └─────────────────┬─────────────────┘  break)
                              │ Yes
            ┌─────────────────▼─────────────────┐  No (escaping
            │ Hoisted setups effect-neutral?     ├─non-idemp. ► ⛔ DECLINE
            │ pure-local / idemp. .to() /        │  write)
            │ *licensed* hasattr init (G1-G4) /  │
            │ lowered assert              [INV3] │
            └─────────────────┬─────────────────┘
                              │ Yes
            ┌─────────────────▼─────────────────┐ Yes
            │ Setup has a lowered assert?         ├──► re-gate ¬cond∨check
            │ compose w/ Transform 3      [INV2] │    (preserve guard)
            └─────────────────┬─────────────────┘
                              │
            ┌─────────────────▼─────────────────┐
            │ Branch value contains a call?      │
            └──────┬──────────────────────┬──────┘
              Yes  │                       │ No
        ┌──────────▼─────────┐   ┌─────────▼─────────┐
        │ torch.cond          │   │ torch.where        │
        │ run only taken path │   │ mux both values    │
        └──────────┬─────────┘   └─────────┬─────────┘
                   └───────────┬────────────┘
                               ▼
                          ✅ TRANSFORM
```

### 4.2 Deferred Side Effects (`print / logger.* → buffer/slot + epilogue flush`)

```
        ┌────────────────────────────────────────────┐
        │ Call tagged side_effect in compiled scope    │
        └─────────────────────┬──────────────────────-┘
            ┌─────────────────▼─────────────────┐  Yes
            │ Name rebound / user-defined?[INV3] ├──────► ⛔ DECLINE
            └─────────────────┬─────────────────┘
                              │ No → builtin/lib
            ┌─────────────────▼─────────────────┐  No
            │ Effect independent of later compute├──────► ⛔ DECLINE
            │ (reorder legal)             [INV3] │
            └─────────────────┬─────────────────┘
                              │ Yes
            ┌─────────────────▼─────────────────┐
            │ Which effect?                      │
            └──────┬──────────────────────┬──────┘
            print  │                      │ logger.* (Logger ∉ graph)
        ┌──────────▼────────┐   ┌─────────▼──────────────────┐
        │ buffer (callee,    │   │ inside a forward() method?  │
        │ args, kwargs)      │   └────┬──────────────────┬────┘
        └──────────┬────────┘    Yes │                  │ No
                   │           ┌──────▼──────┐   ┌───────▼────────┐
                   │           │ slot-register│   │ buffer+flush    │
                   │           │ @load; emit  │   │ (relocates break│
                   │           │ slot+const   │   │ but preserves   │
                   │           │ args; forward│   │ the log)        │
                   │           │ -hook flush  │   └───────┬────────┘
                   │           └──────┬──────┘            │
                   └──────────────────┼────────────────────┘
            ┌─────────────────────────▼─────────────────┐
            │ Hoist return value → flush is trailing      │
            │ epilogue: outputs unchanged, in-order [INV3]│
            └─────────────────────────┬─────────────────-┘
                                      ▼
                                 ✅ TRANSFORM
```

#### 4.2.1 Exceptional exits

Deferral moves *when* an effect is emitted, so an exit path that never reaches
the flush turns deferred output into **lost** output -- an observable change,
and exactly what INV3 forbids. The two mechanisms differ here:

| path | normal exit | exceptional exit |
|---|---|---|
| `logger.*` in a `forward` (slot + forward hook) | hook flushes | **flushes** -- hook registered `always_call=True` |
| `print` / non-forward (local buffer + trailing flush) | trailing flush | **does not flush** -- open gap |

For the hook path, `always_call=True` makes PyTorch run the hook even when
`forward` raises, so buffered calls are replayed in FIFO order before the
exception propagates. Covered by
`test_deferred_logger_flushes_on_exceptional_exit` and
`test_flush_hook_registered_with_always_call`.

**Why the buffered path is not fixed with `try`/`finally`.** The natural fix --
wrapping the body so the flush sits in a `finally` -- is not available. The flush
is deliberately untraceable (it performs the real I/O), and TorchDynamo 2.12
responds to an untraceable call inside a `finally` by abandoning the **entire
frame**: the function compiles to zero FX graphs instead of one, i.e. full eager
fallback. Measured on torch 2.12 on the `side_effect*` fixtures (1 graph → 0).
That trades the graph break being removed for something strictly worse, so it is
rejected.

Closing this gap requires the buffer to outlive the frame: a module-level buffer
plus an eager boundary guard wrapping the compiled callable, which is the shape
`[Trap]` already uses for `__jac_trap_guard__`. Until then the buffered path
preserves output on normal and early-return exits only, and that limit is stated
rather than implied.

### 4.3 Predicated Trap Lowering (`if not C: raise → torch._assert_async`)

```
        ┌────────────────────────────────────────────┐
        │ IfStmt tagged val_guard                      │
        │ (data-dependent cond, body = only raise)     │
        └─────────────────────┬──────────────────────-┘
            ┌─────────────────▼─────────────────┐  No
            │ Shape is `if not C: raise`? [INV1] ├──────► ⛔ DECLINE
            └─────────────────┬─────────────────┘
                              │ Yes  (assertion: C must hold)
            ┌─────────────────▼─────────────────┐
            │ Is C a tensor-producing boolean?   │
            └──────┬──────────────────────┬──────┘
          equal() │                       │ .all/.any/.allclose
          (py bool)│                      │ (already tensor bool)
        ┌──────────▼─────────┐            │
        │ shapes & dtypes     │  No        │
        │ match? static       ├──► assert  │
        │ py-bool guard   [≡] │   False    │
        └──────────┬─────────┘            │
                   │ Yes                   │
        ┌──────────▼─────────┐            │
        │ rebuild (a==b).all()│            │
        │ as tensor op    [≡] │            │
        └──────────┬─────────┘            │
                   └──────────┬────────────┘
            ┌─────────────────▼─────────────────┐  No (not a
            │ Convertible to tensor-bool assert? ├─tensor bool)► ⛔ DECLINE
            │                             [INV4] │
            └─────────────────┬─────────────────┘
                              │ Yes
            ┌─────────────────▼─────────────────┐
            │ Emit torch._assert_async(C, msg)   │
            │ • nested in predicated branch →    │
            │   re-gated later by Transform 1    │
            │ • in @torch.compile fn + literal   │
            │   msg → fold [[GM-TRAP type]]       │
            │   marker + prepend @trap_guard,     │
            │   which restores the type at the    │
            │   async failure boundary            │
            └─────────────────┬─────────────────┘
                              ▼
                         ✅ TRANSFORM
```

---

## 5. Formal algorithms

In the paper's Algorithm-1 style. `⊥` = decline.

### Algorithm 2 -- Predicated Control-Flow Legality & Rewrite

```
Input : IfStmt n tagged dyn_ctrl_fl;  Output: predicated dataflow, or ⊥
 1  c ← cond(n); T ← body(n); F ← elseBody(n)
 2  if F = ∅ ∨ n has elif successors           then return ⊥        ▷ INV1 two-way join
 3  if ¬Materializable(c) ∨ ¬Pure(c)            then return ⊥        ▷ INV1
 4  ⟨K, vT, vF⟩ ← Reconcile(T, F)                                    ▷ INV4 join descriptor
 5  if K = ⊥                                     then return ⊥        // outputs not mergeable
 6  if ¬TensorVal(vT) ∨ ¬TensorVal(vF)          then return ⊥        ▷ INV4 None/non-tensor
 7  ST ← T∖{last(T)};  SF ← F∖{last(F)}                              // hoisted setups
 8  foreach (S, taken) ∈ {(ST,⊤),(SF,⊥)} do
 9      if ¬CtrlFree(S)                          then return ⊥        ▷ INV2 path totality
10      foreach s ∈ S do
11          if IsEqAssert(s)                     then return ⊥        // cannot re-gate (Alg.4)
12          else if IsLoweredAssert(s)           then s ← Regate(s,c,taken)  ▷ INV2 compose
13          else if ¬EffectNeutral(s)            then return ⊥        ▷ INV3 escape/idempotency

EffectNeutral(s) ≡                                   // licensed-safe hoist forms only
   PureLocalWrite(s)            // s: ‹x̄ ← e›, x̄ all bare locals, ¬HasCall(e)
 ∨ DeviceMove(s)               // s: ‹x ← r.to(..)›, x ≡ r  (idempotent self-write)
 ∨ ExistenceGuardedInit(s)     // see below -- the `hasattr` *shape* alone is NOT enough
   // everything else (escaping attr/subscript writes, unresolved calls) ⇒ false

ExistenceGuardedInit(s) ≡                            // s: ‹if ¬hasattr(X,"k"): B›
   NoElse(s)                                         // G1 shape: an init guard has no alternative
 ∧ ∃ b ∈ B : Writes(b, X.k)                          // G2 the guard closes ⇒ replay is a no-op
 ∧ ∀ b ∈ B : Writes(b, X.k) ∨ PureLocalWrite(b) ∨ DeviceMove(b)   // G3 only the init may call
 ∧ Confined(X.k, region)                             // G4 no read of X.k outside the region
14  p ← fresh();  emit  p ← c                                        // hoist predicate
15  emit Hoist(ST); emit Hoist(SF)
16  if HasCall(vT) ∨ HasCall(vF) then  sel ← cond(p, λ.vT, λ.vF, ()) // run only taken path
17  else                               sel ← where(p, vT, vF)        // mux both values
18  emit  Realize(K, sel);   remove n

Reconcile(T,F):                                       // join-compatibility analysis
   if last(T)=‹x ← a›, last(F)=‹x ← b›          → ⟨assign(x), a, b⟩
   if last(T)=‹return a›, last(F)=‹return b›     → ⟨return, a, b⟩
   if last(T)=‹g(ā)›, last(F)=‹g(b̄)›, kwargs(T)=kwargs(F), |ā|=|b̄|
                                                → ⟨call(g,kwargs), ā, b̄⟩
                                                  // Realize muxes each differing arg i: where(p,aᵢ,bᵢ)
   otherwise                                    → ⊥
Regate(assert(C,m), c, taken):
   return assert( taken ? (¬c ∨ C) : (c ∨ C), m )
```

### Algorithm 3 -- Side-Effect Deferral Legality & Rewrite

```
Input : Call n tagged side_effect, in region R;  Output: deferred effect, or ⊥
 1  f ← callee(n)
 2  if ¬BindingStable(n)             then return ⊥                   ▷ INV3 binding stability
 3  if ¬Independent(n, succ(n))      then return ⊥                   ▷ INV3 reorder legality
 4  if f = print then
 5      emit  buf.append(⟨f, snapshot(args(n)), kwargs(n)⟩)          // value-replayable, cloned
 6  else if IsLogger(f) then                                        // Logger object ∉ graph
 7      if n ∈ forward-method M then
 8          sl ← RegisterSlot(f) @ load(M)                           // bound method out of graph
 9          emit  log_emit(sl, args(n), kwargs(n))                   // int slot + const args
10          ensure  M.register_forward_hook(flush)                  // replay post-graph
11      else
12          emit  buf.append(⟨f, snapshot(args(n)), kwargs(n)⟩)     // relocates, log preserved
13  remove n;  mark enclosingFn as needs-flush
─── once per enclosing fn ───
14  HoistReturnValue()              // r ← expr; flush(); return r   ▷ INV3 output-neutral
15  emit  flush()  as trailing epilogue                             // in original order

snapshot(args):                                       // INV3 value-snapshot at call site
   foreach non-constant aᵢ ∈ args:                    // literals are immutable, left as-is
      tᵢ ← fresh();  hoist  tᵢ ← aᵢ  before the append // bind once, original eval order
      replace aᵢ with  (tᵢ.clone() if hasattr(tᵢ,'clone') else tᵢ)
   // hasattr resolves statically under Dynamo ⇒ clone is a native in-graph op for
   // tensors, a no-op otherwise; protects the deferred value from a later in-place
   // mutation of the same tensor (paper Table 1, clone(args)).
```

### Algorithm 4 -- Predicated Trap-Lowering Legality & Rewrite

```
Input : IfStmt n tagged val_guard;  Output: graph-native assertion, or ⊥
 1  if |body(n)| ≠ 1 ∨ body(n) ≠ ‹raise›   then return ⊥            ▷ INV1
 2  c ← cond(n)
 3  if c ≠ ¬X  (unary not)                  then return ⊥            ▷ INV1 (assert X must hold)
 4  m ← message(body(n))
 5  if X = torch.equal(a,b) then                                     // returns Python bool
 6      g ← (shape(a)=shape(b) ∧ dtype(a)=dtype(b))                  ▷ [≡] static guard, no break
 7      C ← g ? (a==b).all() : tensor(False)                         // value-equiv of equal
 8  else if TensorBool(X) then  C ← X                                // already tensor bool
 9  else                        return ⊥                             ▷ INV4 not tensor-bool
10  e ← exceptionType(n)                                            // e.g. ValueError
11  if n ∈ a @torch.compile fn ∧ m is a literal then               // type-preservation scope
12      m ← "[[GM-TRAP "+e+"]]"+m+"[[/GM-TRAP]]"                    // fold marker at compile time
13      mark enclosingFn → prepend @trap_guard (boundary restore)
14  emit  _assert_async(C, m);   remove n
15  // C is a value ⇒ if n nested in a dyn_ctrl_fl branch, Alg.2:12 re-gates C  ▷ INV2 seam
16  // A graph-native assert fails ASYNC at the call boundary, so the type is
17  // restored there (not in-source): @trap_guard catches the RuntimeError and
18  // re-raises e(m). Module/forward entries have no decorator ⇒ message-only.
```

### Algorithm 5 -- Driver (scheduling = the composition guarantee)

```
Input : Module U with ≥1 torch.compile entry point
 1  Detect(U)                          // Alg. 1 → tags {dyn_ctrl_fl, val_guard, side_effect}
 2  foreach n tagged val_guard   in post-order do  Alg.4(n)   // ── trap BEFORE predicate
 3  foreach n tagged dyn_ctrl_fl in post-order do  Alg.2(n)   //    so nested guards are
 4  foreach n tagged side_effect in post-order do  Alg.3(n)   //    values when Alg.2 hoists
```

The order in lines 2-3 is the formal statement of the **compound-break** answer:
Alg. 4 turns an inner guard into a value-level `_assert_async` *before* Alg. 2
reaches the enclosing branch, so Alg. 2 line 12 (`Regate`) can lift it
predicate-safely. Post-order resolves inner regions before their enclosers.

---

## 6. Soundness

> **Claim.** For every region transformed by Algorithms 2-4, the rewritten program
> `P'` is observationally equivalent to `P` on the input domain: identical tensor
> outputs, identical observable effects in original program order, and an
> assertion fires in `P'` iff the original guard would fire in `P`.
>
> **Observability domain.** "Observable" here means: values returned, effects
> performed (I/O, log output, non-idempotent state writes), exceptions raised,
> and any state read *by code within the compiled module*. It does **not** cover
> an external observer reading the object's attribute dictionary directly --
> see the memoization caveat in §6.1.
>
> **Argument.** Each algorithm reaches its `emit` only after discharging the
> invariants that establish this equality for its rewrite (INV1 predicate
> well-formedness, INV2 path totality + re-gating, INV3 effect recoverability +
> binding stability, INV4 join compatibility). Every primitive is a conservative
> must-predicate, so no path to `emit` rests on an unproven property. Otherwise
> the algorithm returns `⊥` and the region is left identical to `P`, where
> equivalence holds trivially. ∎

### 6.1 Caveat -- speculated memoization is a state change, not a no-op

`ExistenceGuardedInit` is the one licensed hoist that **creates state earlier
than `P` would**. In the Phi-4 case (Fig. 3), predication runs
`if not hasattr(self,'long_inv_freq'): self.long_inv_freq = rope_init_fn(...)`
unconditionally, so on an execution where the original predicate is always false
`P` never creates `long_inv_freq` and `P'` always does. Calling this hoist
"idempotent" is precise about *replay* (running it twice equals running it once)
but not about *speculation* (running it at all). The two are different
obligations, and only the first follows from the `hasattr` guard.

What makes the hoist legal is therefore not idempotency but **confinement**
(G4): the attribute is a private memoization slot whose existence no other code
in the module can test or read, so no observer inside the observability domain
can distinguish "cached now" from "cached on a later call". G2 and G3 then
ensure the slot is genuinely closed by the write and that nothing else rides
along in the guard body. When G4 fails -- another method reads the attribute, or
a `hasattr`/`getattr` elsewhere names it -- the rewrite is declined.

Two assumptions remain outside what the analysis can discharge, and are stated
rather than proved:

1. **Cross-module observers.** Confinement is checked over the compiled module.
   An external `hasattr`, a `__dict__`/`state_dict()`/`named_buffers()` walk, or
   a subclass in another file can still observe the attribute existing early.
   For a memoization slot (not a registered buffer or parameter) this is
   invisible to normal model use, but it is not a proof.
2. **Initializer purity.** The initializer call in the memoizing write is not
   resolved and validated -- unlike calls in the *branch value*, which Alg. 2
   requires to resolve through UniiR. If it performs I/O, consumes RNG, or
   mutates global state, speculation makes that happen on runs `P` avoided.
   Resolving the callee and requiring purity is the natural way to close this;
   until then it is a licensed-form assumption, and the honest phrasing of the
   guarantee is *equivalence modulo private memoization state*, not
   unconditional equivalence.

---

## 7. Mapping to the implementation

| Concept | Where |
|---------|-------|
| Detection / tagging (Alg. 1) | `graph_break_detect_pass.jac` |
| Alg. 2 (predicate ctrl-flow) | `predicate_ctrl_flow_pass.jac` -- `_setups_safe` (INV2 control + INV3 effect), `_effect_neutral`/`_is_existence_guard`/`_is_device_move`/`_all_local_targets` (idempotent-write licensing & escape), `_has_none` (INV4), `Reconcile` ≈ `_same_lhs`/`_merge_common_call`, `_use_cond` (where vs cond), `_gate_asserts`/`Regate` (INV2 compose), `_is_eq_assert` bail |
| Alg. 3 (defer side effects) | `defer_side_effect_pass.jac` -- `_is_user_defined` (INV3 binding), `_buffer_entry`/`_clone_guard` (snapshot: hoist + clone-if-tensor), forward-hook slot mechanism, return-value hoist |
| Alg. 4 (trap lowering) | `trap_lowering_pass.jac` -- `if not C: raise` match, `torch.equal` → `__jac_tensor_eq_assert__` with shape/dtype guard, tensor-bool direct; `_marked_message` (fold exception type into a literal marker), `_enclosing_compiled_fn`/`exit_ability` (prepend `@__jac_trap_guard__`); `runtime.impl.jac` `trap_guard` (boundary restore) |
| Alg. 5 (scheduling) | `jac0core/compiler.jac` `get_py_code_gen` -- order: Detect → Trap → Predicate → DeferSideEffect |

### Tests

- Transformation-level (no torch): `test_graphmend_{detect,ctrl_flow,side_effect,trap_lowering,py_support}.jac`
- Runtime defragmentation + semantics (torch-gated): `test_graphmend_integration.py`
- Compound break (INV2 composition): fixture `fixtures/graphmend/nested_guard_in_branch.jac`;
  `test_nested_guard_in_branch_defragmented` (graph 1) and
  `test_nested_guard_preserves_conditional_assert_semantics` (untaken-branch guard
  stays dormant).

### Known limitations (honest scope)

- **[Defer] on exceptional exits: fixed for logger, open for print.** The
  forward-hook (logger-in-`forward`) path flushes on a raising forward because
  the hook is registered `always_call=True`. The buffered `print` / non-forward
  path still flushes only on normal and early-return exits, so an exception
  escaping that function drops its buffered output. A `try`/`finally` cannot fix
  it -- an untraceable call in a `finally` makes Dynamo 2.12 abandon the frame
  (1 FX graph becomes 0). See 4.2.1 for the measurement and the boundary-guard
  design that would close it.
- **Exception type -- restored within scope.** The original exception type and
  message are restored when the guard sits in a `@torch.compile`-decorated
  function and the message is a string literal: the type is folded into a marker
  and an eager `@__jac_trap_guard__` boundary decorator re-raises it. Out of
  scope (still message-only as a `RuntimeError`): guards in `nn.Module` forward
  entries compiled at a call site (no decorator to attach), non-literal messages,
  and custom non-builtin exception types (these fall back to `RuntimeError` so
  the message is never lost). Because the assert is graph-native and fails
  asynchronously, the error surfaces at the next sync/call boundary rather than
  at the original guard line.
- **`torch.equal` nested in a predicated branch** is declined (its check is
  computed inside the helper and cannot be re-gated). Only the tensor-bool guard
  form composes.
- **Multi-statement same-LHS assignment** branches: predication's multi-statement
  path currently handles only a shared *trailing call* (`Reconcile` call case), so
  compound fixtures must end in a shared `ExprStmt` call.
- **Speculated memoization creates private state early.** The one hoist that is
  not effect-free in the strict sense; legal by confinement, not by idempotency.
  See §6.1 for the two residual assumptions (cross-module observers, initializer
  purity).
- **Hoist effect-neutrality is conservative.** Setups are hoisted only when proven
  neutral by the licensed forms in `EffectNeutral` (pure-local write, idempotent
  `.to()` device move, `hasattr`-guarded init, re-gated lowered assert). A setup
  with an escaping attribute/subscript write, or a call we do not resolve, is
  declined even if it would in fact be safe -- coverage is traded for soundness,
  so an unsafe non-idempotent write (e.g. `self.counter += 1`) is never hoisted.

```
