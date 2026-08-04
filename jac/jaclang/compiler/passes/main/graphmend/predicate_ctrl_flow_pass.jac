"""GraphMend: Predicated Data-dependent Control Flow transformation.

Rewrites data-dependent if/else branches (tagged `dyn_ctrl_fl` by
GraphBreakDetectPass) into a single `torch.where` expression so both paths stay
inside one FX graph, eliminating the graph break (paper Figs. 1-2).

Two reconciliation shapes are handled conservatively -- only when both branches
produce the *same* output, so the rewrite is provably semantics-preserving:

  if c { x = A; } else { x = B; }      ->  __gm_cond = c;
                                           __gm_true = A;
                                           __gm_false = B;
                                           x = torch.where(__gm_cond, __gm_true, __gm_false);

  if c { return A; } else { return B; } ->  ... ; return torch.where(...);

When a branch value contains a function call, `torch.where` is the wrong tool:
it evaluates BOTH branches, so an asymmetric pair like `x = f(a)` / `x = g(b)`
would run f AND g (wasteful, and unsafe if a branch is only valid when taken).
For that case we emit `torch.cond`, a graph-native conditional that executes only
the selected branch:

  if c { x = f(a); } else { x = g(b); } ->  __gm_cond = c;
                                            x = torch.cond(__gm_cond,
                                                           lambda : f(a), lambda : g(b), ());

`torch.cond` requires a scalar predicate (true for genuine data-dependent breaks)
and both branches to return matching tensor structures. If a branch yields a
non-tensor (e.g. `None`), no transform applies and the break is left intact.

Multi-statement branches sharing a trailing call hoist their setups to run
unconditionally, but only when each setup is provably effect-neutral on the
untaken path (pure-local write, idempotent `.to()` device move, licensed
`hasattr`-guarded init, or a re-gated lowered assert) -- see
`_setups_safe`/`_effect_neutral`. A non-idempotent observable write (e.g.
`self.counter += 1`) is declined, leaving the break intact rather than hoisting
it unsoundly.

The `hasattr`-guarded init is the one licensed hoist that creates state earlier
than the original would; the `hasattr` *shape* alone does not make it safe. See
`_is_existence_guard` for the four conditions (memoizing write, neutral body,
module-scope confinement) and SEMANTIC_SAFETY.md §6.1 for what remains an
assumption.

Anything more complex (differing targets, deeper multi-statement branches, elif
chains) is left untouched. Gated by --graphmend.
"""
import ast as ast3;
import jaclang.jac0core.unitree as uni;
import from jaclang.jac0core.passes { UniPass }
import from jaclang.jac0core.constant { Tokens as Tok }

"""Rewrite data-dependent if/else into torch.where."""
obj PredicateCtrlFlowPass(UniPass) {
    has _enabled: bool = False,
        _counter: int = 0;

    """Gate the pass on the --graphmend flag."""
    def before_pass {
        self._enabled = getattr(self.prog._compile_options, 'graphmend', False);
        if not self._enabled {
            self.prune_signal = True;
        }
    }

    """Transform a tagged data-dependent conditional on exit."""
    def exit_if_stmt(nd: uni.IfStmt) -> None {
        if not self._enabled {
            return;
        }
        if (nd?._gm_break_kind != 'dyn_ctrl_fl') {
            return;
        }
        eb = nd.else_body;
        if (eb is None) or (not isinstance(eb, uni.ElseStmt)) {
            return;
        }
        tb = nd.body;
        fb = eb.body;
        if (not tb) or (not fb) {
            return;
        }
        # Single-statement branches: same-target assign, same return, or a
        # common call (no setup statements to hoist).
        if (len(tb) == 1) and (len(fb) == 1) {
            a0 = tb[0];
            b0 = fb[0];
            if isinstance(a0, uni.Assignment) and isinstance(b0, uni.Assignment) {
                lhs = self._same_lhs(a0, b0);
                if (lhs is not None)
                and (a0.value is not None)
                and (b0.value is not None)
                and (not self._has_none(a0.value, b0.value)) {
                    self._rewrite_assignment(nd, lhs, a0.value, b0.value);
                }
            } elif isinstance(a0, uni.ReturnStmt) and isinstance(b0, uni.ReturnStmt) {
                if (a0.expr is not None)
                and (b0.expr is not None)
                and (not self._has_none(a0.expr, b0.expr)) {
                    self._rewrite_return(nd, a0.expr, b0.expr);
                }
            } elif isinstance(a0, uni.ExprStmt) and isinstance(b0, uni.ExprStmt) {
                self._merge_common_call(nd, a0.expr, b0.expr, [], []);
            }
            return;
        }
        # Multi-statement branches: both must END in the same call (paper Fig. 4).
        # The preceding setup statements are hoisted to run unconditionally
        # (predication computes both paths), provided they carry no control flow.
        a_last = tb[-1];
        b_last = fb[-1];
        if isinstance(a_last, uni.ExprStmt) and isinstance(b_last, uni.ExprStmt) {
            if self._setups_safe(tb[:-1], nd) and self._setups_safe(fb[:-1], nd) {
                self._merge_common_call(nd, a_last.expr, b_last.expr, tb[:-1], fb[:-1]);
            }
        }
    }

    """True if setup statements can be safely hoisted out of the branch.

    A setup runs UNCONDITIONALLY after the rewrite (predication computes both
    paths), so it must be observationally neutral on the untaken path. Two
    obligations (paper Table 1 / Fig. 3 Phi-4 legality):

      1. Control neutrality -- no return/raise/break/continue, else hoisting
         would make a control transfer unconditional.
      2. Effect neutrality -- any write must be unobservable when run on the
         untaken path: see `_effect_neutral`. This rejects non-idempotent
         observable writes (e.g. `self.counter += 1`) that the old control-only
         check would have hoisted unsoundly.
    """
    def _setups_safe(setups: list, region: uni.IfStmt) -> bool {
        for s in setups {
            if isinstance(s, (uni.ReturnStmt, uni.RaiseStmt, uni.CtrlStmt)) {
                return False;
            }
            if s.get_all_sub_nodes(uni.ReturnStmt) {
                return False;
            }
            if s.get_all_sub_nodes(uni.RaiseStmt) {
                return False;
            }
            if s.get_all_sub_nodes(uni.CtrlStmt) {
                return False;
            }
            if not self._effect_neutral(s, region) {
                return False;
            }
        }
        return True;
    }

    """True if hoisting `s` to run on both paths changes nothing observable.

    Licensed forms (anything else is conservatively declined):
      - a lowered validation guard `torch._assert_async(...)` -- re-gated onto the
        branch predicate afterwards (`_gate_asserts`), so it stays conditional;
      - an existence-guarded initialization `if not hasattr(X, "k"): ...` -- the
        guard makes re-execution a no-op (paper's idempotent init);
      - an assignment that is either a pure write to local name(s) (the value is
        recomputed, not an escaping effect) or an idempotent device move
        `x = x.to(...)` (writing a tensor back to itself).
    """
    def _effect_neutral(s: uni.UniNode, region: uni.IfStmt) -> bool {
        if self._is_assert_async(s) {
            return True;
        }
        if self._is_existence_guard(s, region) {
            return True;
        }
        if isinstance(s, uni.Assignment) {
            return self._assignment_neutral(s);
        }
        return False;
    }

    """True if `s` is a *licensed* existence-guarded initialization.

    The `hasattr` shape alone is NOT sufficient. Speculating the guard body makes
    the initialized state exist on runs where the original program never created
    it, so four conditions must hold for that state change to be unobservable:

      G1. shape -- `if not hasattr(<recv>, "<literal>")` with no `else` branch;
      G2. memoization -- the body definitely assigns `<recv>.<literal>`, so the
          guard flips to true and re-execution is a genuine no-op. Without this
          the guard never closes and the body re-runs on every call;
      G3. body neutrality -- every other statement in the body is itself
          effect-neutral. Only the memoizing assignment may contain a call (the
          initializer), matching the paper's "initialization guarded by its own
          existence check" license. Arbitrary I/O or escaping writes smuggled in
          under a `hasattr` header are declined;
      G4. confinement -- the initialized attribute is read nowhere in the module
          outside this branch region (`_attr_confined`), so its earlier existence
          cannot be observed by any other code we compile.

    G4 is a module-scope check: state escaping to *other* modules (an external
    `hasattr`, `__dict__`/`state_dict` inspection, a subclass) is outside what
    this analysis can see, and is the residual assumption behind the hoist.
    """
    def _is_existence_guard(s: uni.UniNode, region: uni.IfStmt) -> bool {
        if not isinstance(s, uni.IfStmt) {
            return False;
        }
        # G1 -- an init guard has no alternative path.
        if s.else_body is not None {
            return False;
        }
        keyed = self._hasattr_key(s.condition);
        if keyed is None {
            return False;
        }
        (recv, key) = keyed;
        memoized = False;
        for st in (s.body or []) {
            if not isinstance(st, uni.Assignment) {
                return False;
            }
            if self._writes_attr(st, recv, key) {
                # G2 -- the memoizing write; its value may contain the
                # initializer call.
                memoized = True;
                continue;
            }
            # G3 -- anything else in the body must stand on its own.
            if not self._assignment_neutral(st) {
                return False;
            }
        }
        if not memoized {
            return False;
        }
        return self._attr_confined(region, key);
    }

    """Return `(receiver_text, key)` for `not hasattr(<recv>, "<literal>")`."""
    def _hasattr_key(c: uni.Expr) -> (tuple | None) {
        if (not isinstance(c, uni.UnaryExpr))
        or (c.op is None)
        or (c.op.value != 'not') {
            return None;
        }
        inner = c.operand;
        if not (
            isinstance(inner, uni.FuncCall)
            and isinstance(inner.target, uni.Name)
            and (inner.target.value == 'hasattr')
        ) {
            return None;
        }
        params = list(inner.params or []);
        if len(params) != 2 {
            return None;
        }
        (recv, key) = (params[0], params[1]);
        # a string literal surfaces as `String` or `MultiString`; both expose
        # the unquoted text via `lit_value`.
        if isinstance(recv, uni.KWPair)
        or (not isinstance(key, (uni.String, uni.MultiString))) {
            return None;
        }
        return (recv.unparse().replace(' ', ''), key.lit_value);
    }

    """True if assignment `s` writes `<recv>.<key>` (possibly in a tuple target)."""
    def _writes_attr(s: uni.Assignment, recv: str, key: str) -> bool {
        for t in (s.target or []) {
            targets = list(t.values) if isinstance(t, uni.TupleVal) else [t];
            for tg in targets {
                if (
                    isinstance(tg, uni.AtomTrailer)
                    and isinstance(tg.right, uni.Name)
                    and (tg.right.value == key)
                    and (tg.target.unparse().replace(' ', '') == recv)
                ) {
                    return True;
                }
            }
        }
        return False;
    }

    """True if attribute `key` is referenced nowhere in the module outside `region`.

    Any `<expr>.key` access or any string literal equal to `key` (which could feed
    `hasattr`/`getattr`/`setattr`) found outside the branch being rewritten means
    the early creation of the attribute is observable, so the hoist is declined.
    """
    def _attr_confined(region: uni.IfStmt, key: str) -> bool {
        mod = region.find_parent_of_type(uni.Module);
        if mod is None {
            return False;
        }
        for at in mod.get_all_sub_nodes(uni.AtomTrailer) {
            if (
                isinstance(at.right, uni.Name)
                and (at.right.value == key)
                and (not self._within(at, region))
            ) {
                return False;
            }
        }
        for typ in [uni.String, uni.MultiString] {
            for st in mod.get_all_sub_nodes(typ) {
                if (st.lit_value == key) and (not self._within(st, region)) {
                    return False;
                }
            }
        }
        return True;
    }

    """True if `nd` sits inside `anc` in the tree."""
    def _within(nd: uni.UniNode, anc: uni.UniNode) -> bool {
        cur = nd;
        while cur is not None {
            if cur is anc {
                return True;
            }
            cur = cur.parent;
        }
        return False;
    }

    """True if assignment `s` is neutral to hoist: an idempotent device move, or a
    pure write to local name target(s) (no call in the value)."""
    def _assignment_neutral(s: uni.Assignment) -> bool {
        if self._is_device_move(s) {
            return True;
        }
        return (
            (s.value is not None)
            and (not self._has_call(s.value))
            and self._all_local_targets(s.target)
        );
    }

    """True if `s` is `lhs = <recv>.to(...)` with `lhs` textually equal to the
    receiver -- a device move that is idempotent (re-running is a no-op)."""
    def _is_device_move(s: uni.Assignment) -> bool {
        v = s.value;
        if (v is None) or (len(s.target) != 1) {
            return False;
        }
        if not (
            isinstance(v, uni.FuncCall)
            and isinstance(v.target, uni.AtomTrailer)
            and isinstance(v.target.right, uni.Name)
            and (v.target.right.value == 'to')
        ) {
            return False;
        }
        recv = v.target.target;
        return (
            recv.unparse().replace(' ', '') == s.target[0].unparse().replace(' ', '')
        );
    }

    """True if every assignment target is a bare local Name (not an attribute or
    subscript), so the write does not escape to observable state."""
    def _all_local_targets(targets: list) -> bool {
        for t in targets {
            if not isinstance(t, uni.Name) {
                return False;
            }
        }
        return len(targets) > 0;
    }

    """Return the common LHS name if both assigns target the same single name."""
    def _same_lhs(a: uni.Assignment, b: uni.Assignment) -> (uni.Name | None) {
        if (len(a.target) != 1) or (len(b.target) != 1) {
            return None;
        }
        ta = a.target[0];
        tb = b.target[0];
        if (
            isinstance(ta, uni.Name)
            and isinstance(tb, uni.Name)
            and (ta.value == tb.value)
        ) {
            return self._gen_name(ta, ta.value, True);
        }
        return None;
    }

    """Mint a synthetic Name anchored at `at`, with Store context if `store`."""
    def _gen_name(at: uni.UniNode, value: str, store: bool) -> uni.Name {
        nm = uni.Name.gen_stub_from_node(at, value);
        if store {
            nm.expr_ctx = uni.ExprCtx.STORE;
        }
        return nm;
    }

    """Build `target = value` as a fresh Assignment."""
    def _assign(self_target: uni.Name, value: uni.Expr) -> uni.Assignment {
        return uni.Assignment(
            target=[self_target],
            value=value,
            type_tag=None,
            mutable=True,
            kid=[self_target, value]
        );
    }

    """Build a `torch.where(cond, t, f)` call anchored at `at`."""
    def _make_where(
        at: uni.UniNode, cond: uni.Expr, t: uni.Expr, f: uni.Expr
    ) -> uni.FuncCall {
        return self._torch_call(at, 'where', [cond, t, f]);
    }

    """Build a `torch.<fname>(*args)` call anchored at `at`."""
    def _torch_call(at: uni.UniNode, fname: str, args: list) -> uni.FuncCall {
        torch_nm = self._gen_name(at, 'torch', False);
        fn_nm = self._gen_name(at, fname, False);
        target = uni.AtomTrailer(
            target=torch_nm,
            right=fn_nm,
            is_attr=True,
            is_null_ok=False,
            kid=[torch_nm, fn_nm]
        );
        return uni.FuncCall(
            target=target, params=args, genai_call=None, kid=[target] + args
        );
    }

    """Hoist condition/branches to temps; return (stmts, cond_ref, t_ref, f_ref)."""
    def _hoist(at: uni.UniNode, cond: uni.Expr, t: uni.Expr, f: uni.Expr) -> tuple {
        n = self._counter;
        self._counter += 1;
        cond_tmp = self._gen_name(at, f"__gm_cond_{n}", True);
        true_tmp = self._gen_name(at, f"__gm_true_{n}", True);
        false_tmp = self._gen_name(at, f"__gm_false_{n}", True);
        stmts = [
            self._assign(cond_tmp, cond),
            self._assign(true_tmp, t),
            self._assign(false_tmp, f)
        ];
        cond_ref = self._gen_name(at, f"__gm_cond_{n}", False);
        true_ref = self._gen_name(at, f"__gm_true_{n}", False);
        false_ref = self._gen_name(at, f"__gm_false_{n}", False);
        return (stmts, cond_ref, true_ref, false_ref);
    }

    """Rewrite same-target assignment branches into torch.where / torch.cond."""
    def _rewrite_assignment(
        nd: uni.IfStmt, lhs: uni.Name, true_e: uni.Expr, false_e: uni.Expr
    ) {
        if self._use_cond(true_e, false_e) {
            (stmts, c_ref) = self._hoist_cond(nd, nd.condition);
            cond_call = self._make_cond(nd, c_ref, true_e, false_e);
            self._replace(nd, stmts + [self._assign(lhs, cond_call)]);
            return;
        }
        (stmts, c_ref, t_ref, f_ref) = self._hoist(nd, nd.condition, true_e, false_e);
        where_call = self._make_where(nd, c_ref, t_ref, f_ref);
        final = self._assign(lhs, where_call);
        self._replace(nd, stmts + [final]);
    }

    """Rewrite same-shape return branches into a torch.where / torch.cond return."""
    def _rewrite_return(nd: uni.IfStmt, true_e: uni.Expr, false_e: uni.Expr) {
        if self._use_cond(true_e, false_e) {
            (stmts, c_ref) = self._hoist_cond(nd, nd.condition);
            cond_call = self._make_cond(nd, c_ref, true_e, false_e);
            ret = uni.ReturnStmt(expr=cond_call, kid=[cond_call]);
            self._replace(nd, stmts + [ret]);
            return;
        }
        (stmts, c_ref, t_ref, f_ref) = self._hoist(nd, nd.condition, true_e, false_e);
        where_call = self._make_where(nd, c_ref, t_ref, f_ref);
        ret = uni.ReturnStmt(expr=where_call, kid=[where_call]);
        self._replace(nd, stmts + [ret]);
    }

    """Prefer torch.cond over torch.where when a branch value contains a call.

    torch.where evaluates both branches; for a call that would run it on the
    untaken path too (wasteful, and unsafe if only valid under the condition).
    torch.cond runs just the selected branch. A bare-name / pure-arithmetic
    branch stays on torch.where (cheap, and avoids cond's stricter constraints)."""
    def _use_cond(true_e: uni.Expr, false_e: uni.Expr) -> bool {
        if isinstance(true_e, uni.Literal) or isinstance(false_e, uni.Literal) {
            return False;
        }
        return self._has_call(true_e) or self._has_call(false_e);
    }

    """True if `e` is or contains a function call."""
    def _has_call(e: uni.Expr) -> bool {
        return isinstance(e, uni.FuncCall) or bool(e.get_all_sub_nodes(uni.FuncCall));
    }

    """True if either branch value is a `None` literal -- no tensor select is
    possible (torch.where can't broadcast None, torch.cond can't return it), so
    such branches are left untransformed (e.g. Qwen's `audios = None` else)."""
    def _has_none(a: uni.Expr, b: uni.Expr) -> bool {
        return isinstance(a, uni.Null) or isinstance(b, uni.Null);
    }

    """Hoist only the predicate to a temp; return ([assign], cond_ref)."""
    def _hoist_cond(at: uni.UniNode, cond: uni.Expr) -> tuple {
        n = self._counter;
        self._counter += 1;
        cond_tmp = self._gen_name(at, f"__gm_cond_{n}", True);
        cond_ref = self._gen_name(at, f"__gm_cond_{n}", False);
        return ([self._assign(cond_tmp, cond)], cond_ref);
    }

    """Build `torch.cond(pred, lambda : t, lambda : f, ())` anchored at `at`."""
    def _make_cond(
        at: uni.UniNode, pred: uni.Expr, t: uni.Expr, f: uni.Expr
    ) -> uni.FuncCall {
        torch_nm = self._gen_name(at, 'torch', False);
        cond_nm = self._gen_name(at, 'cond', False);
        target = uni.AtomTrailer(
            target=torch_nm,
            right=cond_nm,
            is_attr=True,
            is_null_ok=False,
            kid=[torch_nm, cond_nm]
        );
        true_fn = uni.LambdaExpr(body=t, signature=None, kid=[t]);
        false_fn = uni.LambdaExpr(body=f, signature=None, kid=[f]);
        operands = uni.TupleVal(
            values=[],
            kid=[at.gen_token(Tok.LPAREN, '('), at.gen_token(Tok.RPAREN, ')')]
        );
        params = [pred, true_fn, false_fn, operands];
        return uni.FuncCall(
            target=target, params=params, genai_call=None, kid=[target] + params
        );
    }

    """Split call params into positional args and a kwarg-name -> KWPair map."""
    def _split_params(params: any) -> tuple {
        pos = [];
        kw = {};
        for p in (params or []) {
            if isinstance(p, uni.KWPair) {
                if p.key is not None {
                    kw[p.key.value] = p;
                }
            } else {
                pos.append(p);
            }
        }
        return (pos, kw);
    }

    """Rewrite branches that call the same function, merging differing args.

    Handles the paper's Fig. 4 reconciliation: when both branches invoke the
    same callee with identical kwargs and the same positional arity, each
    differing positional argument is merged with torch.where and the call is
    emitted once. Any per-branch setup statements (`true_setups`/`false_setups`)
    are hoisted to run unconditionally before the merged call -- e.g.

        if c { x = A(); o.reg("k", x); } else { y = B(); o.reg("k", y); }
    becomes
        __gm_cond = c;
        x = A();
        y = B();
        o.reg("k", torch.where(__gm_cond, x, y));
    """
    def _merge_common_call(
        nd: uni.IfStmt,
        ca: uni.Expr,
        cb: uni.Expr,
        true_setups: list,
        false_setups: list
    ) -> None {
        if (not isinstance(ca, uni.FuncCall)) or (not isinstance(cb, uni.FuncCall)) {
            return;
        }
        if (
            ca.target.unparse().replace(' ', '') != cb.target.unparse().replace(' ', '')
        ) {
            return;
        }
        (a_pos, a_kw) = self._split_params(ca.params);
        (b_pos, b_kw) = self._split_params(cb.params);
        if len(a_pos) != len(b_pos) {
            return;
        }
        if set(a_kw.keys()) != set(b_kw.keys()) {
            return;
        }
        for k in a_kw {
            if a_kw[k].value.unparse() != b_kw[k].value.unparse() {
                return;
            }
        }
        diff = [];
        for i in range(len(a_pos)) {
            if a_pos[i].unparse() != b_pos[i].unparse() {
                diff.append(i);
            }
        }
        if not diff {
            return;
        }
        # A torch.equal guard (`__jac_tensor_eq_assert__`) hoisted out of a branch
        # cannot be re-gated on the predicate (its check is computed inside the
        # helper), so we decline rather than assert it unconditionally.
        for s in (list(true_setups) + list(false_setups)) {
            if self._is_eq_assert(s) {
                return;
            }
        }
        n = self._counter;
        self._counter += 1;
        cond_tmp = self._gen_name(nd, f"__gm_cond_{n}", True);
        cond_assign = self._assign(cond_tmp, nd.condition);
        merged = [];
        for i in range(len(a_pos)) {
            if i in diff {
                cref = self._gen_name(nd, f"__gm_cond_{n}", False);
                merged.append(self._make_where(nd, cref, a_pos[i], b_pos[i]));
            } else {
                merged.append(a_pos[i]);
            }
        }
        new_params = merged;
        for k in a_kw {
            new_params.append(a_kw[k]);
        }
        new_call = uni.FuncCall(
            target=ca.target,
            params=new_params,
            genai_call=None,
            kid=[ca.target] + new_params
        );
        call_stmt = uni.ExprStmt(expr=new_call, in_fstring=False, kid=[new_call]);
        gated_true = self._gate_asserts(list(true_setups), n, True);
        gated_false = self._gate_asserts(list(false_setups), n, False);
        new_nodes = [cond_assign] + gated_true + gated_false + [call_stmt];
        self._replace(nd, new_nodes);
    }

    """True if `stmt` is a graph-native `torch._assert_async(...)` produced by
    the trap-lowering pass (a lowered tensor-bool validation guard)."""
    def _is_assert_async(stmt: uni.UniNode) -> bool {
        if not isinstance(stmt, uni.ExprStmt) {
            return False;
        }
        call = stmt.expr;
        if not isinstance(call, uni.FuncCall) {
            return False;
        }
        t = call.target;
        return (
            isinstance(t, uni.AtomTrailer)
            and isinstance(t.right, uni.Name)
            and (t.right.value == '_assert_async')
            and isinstance(t.target, uni.Name)
            and (t.target.value == 'torch')
        );
    }

    """True if `stmt` wraps a `__jac_tensor_eq_assert__(...)` call (the torch.equal
    guard form), whose check is computed inside the helper and so cannot be
    re-gated on a branch predicate."""
    def _is_eq_assert(stmt: uni.UniNode) -> bool {
        if not isinstance(stmt, uni.ExprStmt) {
            return False;
        }
        call = stmt.expr;
        return (
            isinstance(call, uni.FuncCall)
            and isinstance(call.target, uni.Name)
            and (call.target.value == '__jac_tensor_eq_assert__')
        );
    }

    """Re-gate hoisted `torch._assert_async` calls onto the branch predicate.

    A guard lowered to `torch._assert_async(check, msg)` and lifted out of a
    predicated branch would otherwise assert unconditionally, changing semantics
    (the check only applied when the branch was taken). We rewrite its condition
    so the assertion is vacuously satisfied on the untaken path:

      true branch  (taken when cond):  check -> torch.logical_or(torch.logical_not(cond), check)
      false branch (taken when ~cond): check -> torch.logical_or(cond, check)

    `taken_when_true` selects which gate to apply. Non-assert setups pass through
    unchanged. `n` indexes the hoisted `__gm_cond_{n}` predicate temp.
    """
    def _gate_asserts(setups: list, n: int, taken_when_true: bool) -> list {
        out = [];
        for s in setups {
            if not self._is_assert_async(s) {
                out.append(s);
                continue;
            }
            call = s.expr;
            pos = [
                p
                for p in (call.params or [])
                if not isinstance(p, uni.KWPair)
            ];
            if not pos {
                out.append(s);
                continue;
            }
            check = pos[0];
            cref = self._gen_name(s, f"__gm_cond_{n}", False);
            if taken_when_true {
                gate = self._torch_call(s, 'logical_not', [cref]);
            } else {
                gate = cref;
            }
            new_check = self._torch_call(s, 'logical_or', [gate, check]);
            new_params = [new_check] + list(call.params[1:]);
            new_call = uni.FuncCall(
                target=call.target,
                params=new_params,
                genai_call=None,
                kid=[call.target] + new_params
            );
            out.append(uni.ExprStmt(expr=new_call, in_fstring=False, kid=[new_call]));
        }
        return out;
    }

    """Replace `old` with `new_nodes` in its parent's typed body and kid list."""
    def _replace(old: uni.UniNode, new_nodes: list) -> None {
        parent = old.parent;
        if parent is None {
            return;
        }
        body = parent?.body;
        if isinstance(body, list) and (old in body) {
            i = body.index(old);
            parent.body = body[:i] + new_nodes + body[(i + 1):];
        }
        if old in parent.kid {
            j = parent.kid.index(old);
            new_kid = list(parent.kid[:j]) + new_nodes + list(parent.kid[(j + 1):]);
            parent.set_kids(new_kid);
        }
        for child in new_nodes {
            child.parent = parent;
        }
    }
}
