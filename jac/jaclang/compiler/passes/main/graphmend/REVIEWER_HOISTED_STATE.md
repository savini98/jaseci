# Reviewer point: hoisted state is created on executions where the original wouldn't

> *"Semantics-preservation claim slightly overstated. Hoisted state (e.g. Phi-4's
> `self.long_inv_freq` in Figure 3) is created on executions where the original
> wouldn't, resulting in different program state vs not having the hoisting
> transformation."*

**Verdict: the reviewer is correct.** The state difference is real, the paper does
not address it, and the word the paper uses ("idempotent") answers a different
question than the one being asked. This document covers only this issue.

---

## 1. The difference, concretely

Phi-4, Figure 3(a). The initialization sits **inside** the data-dependent branch:

```python
if seq_len > original_max_position_embeddings:
    if not hasattr(self, "long_inv_freq"):
        self.long_inv_freq, _ = self.rope_init_fn(...)     # only runs when branch taken
    self.register_buffer("inv_freq", self.long_inv_freq, persistent=False)
else:
    self.original_inv_freq = self.original_inv_freq.to(device)
    self.register_buffer("inv_freq", self.original_inv_freq, persistent=False)
```

Figure 3(b) hoists that initialization **out** of the branch, so it runs
unconditionally:

```python
cond = seq_len > original_max_position_embeddings
if not hasattr(self, "long_inv_freq"):
    self.long_inv_freq, _ = self.rope_init_fn(...)         # now runs on every call
self.original_inv_freq = self.original_inv_freq.to(device)
inv_freq = torch.where(cond, self.long_inv_freq, self.original_inv_freq)
self.register_buffer("inv_freq", inv_freq, persistent=False)
```

Consider a program where `cond` is false on every execution -- short sequences
only, which is the common case.

| | is `self.long_inv_freq` created? |
|---|---|
| original | **no** -- the branch is never taken |
| transformed | **yes** -- the guard is outside the branch now |

The object ends up carrying an attribute it would not otherwise have. That is a
change in program state, and the paper's Section 4.4 claim that GraphMend
"preserves the observable behavior of the program: tensor outputs, **visible
state**, console and log output ... and raised exceptions" does not account for
it.

## 2. Why "idempotent" does not answer the objection

The paper licenses this hoist by calling the write idempotent -- "an
initialization guarded by its own existence check" (Section 4.3, Table 1).

`hasattr` does make the statement **safe to repeat**: running it twice has the
same effect as running it once, because the second run finds the attribute
already there and does nothing.

But that is not what the reviewer asked. There are two separate properties:

- **Repeat safety** -- running it twice equals running it once. `hasattr` gives
  you this.
- **Speculation safety** -- running it *at all*, on a path the original never
  took, changes nothing. `hasattr` does **not** give you this.

The objection is about the second one. A guard that makes re-execution a no-op
says nothing about the very first execution, and the first execution is exactly
what the hoist adds. So the paper's justification is true but does not cover the
case being questioned.

## 3. Why outputs were still identical in the evaluation

`long_inv_freq` is a **cache**. It is written by that one initialization and read
by that one function. Nothing else in the model looks at it. So creating it
earlier than the original would cannot change any value the model computes,
which is why every correctness check in Section 5.1 passed.

But "nothing else reads it" is a property of this particular model. The paper
does not check it, so the safety was accidental rather than established. That is
the actual gap.

## 4. The condition that does justify the hoist

The correct justification is **confinement**, not idempotency:

> The attribute is a private cache. Nothing outside the rewritten region reads
> it or tests for its existence. Therefore no code can distinguish "cached now"
> from "cached on a later call," and the outputs and effects of the program are
> unchanged.

This is a checkable condition, and it is now checked. `_is_existence_guard` in
`predicate_ctrl_flow_pass.jac` requires four things before hoisting an
existence-guarded initialization:

1. **Shape** -- the statement is `if not hasattr(X, "k")` with no `else` branch.
2. **Memoization** -- the body actually assigns `X.k`, so the guard closes and
   later calls really do skip it.
3. **Body neutrality** -- every other statement in the body is independently safe
   to run speculatively. Only the memoizing assignment may contain a call.
4. **Confinement** -- attribute `k` is referenced nowhere in the module outside
   the region being rewritten, including as a string literal that
   `hasattr`/`getattr`/`setattr` could consume.

If any of the four fails, the hoist is declined and the graph break is left in
place. Phi-4 satisfies all four and still transforms exactly as before.

## 5. What remains an assumption

Two things are stated rather than proved, and should be presented that way:

1. **Observers outside the module.** Confinement is checked over the compiled
   module. Code in another file -- an external `hasattr`, a `state_dict()` or
   `__dict__` walk, a subclass -- could still see the attribute existing earlier
   than it otherwise would. For a private cache (not a registered buffer or
   parameter) this is invisible in normal use, but it is not a proof.

2. **The initializer's own behavior.** The hoisted call (`self.rope_init_fn`) is
   not inspected. If it performed I/O or consumed randomness, hoisting would make
   that happen on executions the original avoided. Requiring the compiler to
   resolve and verify this call would close the argument, but the callee is
   looked up dynamically and cannot be resolved -- so that requirement would make
   Phi-4 untransformable and remove it from the results.

## 6. Why local temporaries are not a better answer

An obvious alternative is to avoid touching `self` at all and hold the value in a
local variable:

```python
_gm_long = self.rope_init_fn(...)[0]                # local, never written to self
_gm_orig = self.original_inv_freq.to(device)
inv_freq = torch.where(cond, _gm_long, _gm_orig)
```

This does remove the attribute, but it is worse overall:

- **It destroys the cache.** A local dies at the end of the call, so
  `rope_init_fn` re-runs on every forward pass instead of once. That is permanent
  added work in steady state -- the opposite of what the paper is optimizing.
- **It does not fix the deeper issue.** `rope_init_fn` still executes on runs the
  original skipped; it now executes on *more* of them, not fewer.
- **It replaces one state difference with another.** On a run where the branch
  *is* taken, the original sets `self.long_inv_freq` and the temporary version
  never does. Hoisting sets it too early; temporaries never set it. Both differ
  from the original, in opposite directions.

Because of the last point, temporaries still require the same confinement check --
they do not make the transformation safe on their own. Where a branch value needs
no caching at all, a local is the right choice and is already what the pass emits;
the memoized-initializer case is not that case.

If the state change had to be removed while keeping the cache, the only way is to
store the cached value in a compiler-owned side table keyed by the object rather
than as an attribute on it. That preserves compute-once and leaves the user's
object untouched, at the cost of extra machinery and an object-lifetime question.

## 7. Recommended wording change

Section 4.4 currently claims preservation of "visible state" without
qualification. Suggested replacement:

> GraphMend preserves tensor outputs, console and log output in content and
> order, and raised exceptions with their type and message. A memoized
> initialization may be populated earlier than in the original program; this is
> permitted only when the analysis establishes that the memoized value is read
> nowhere outside the rewritten region, so no observer can distinguish the two
> schedules.

Table 1's `[Where]` row should also be reconciled with §5.2 above: it states that
every call in the region resolves through UniiR and is validated free of
observable effects, which is not true of the initializer call in the guard body --
the call appearing in Figure 3, the paper's own worked example.
