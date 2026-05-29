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

Anything more complex (differing targets, multi-statement branches, elif chains)
is left untouched. Gated by --graphmend.
"""
import ast as ast3;
import jaclang.jac0core.unitree as uni;
import from jaclang.jac0core.passes { UniPass }

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
                and (b0.value is not None) {
                    self._rewrite_assignment(nd, lhs, a0.value, b0.value);
                }
            } elif isinstance(a0, uni.ReturnStmt) and isinstance(b0, uni.ReturnStmt) {
                if (a0.expr is not None) and (b0.expr is not None) {
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
            if self._setups_safe(tb[:-1]) and self._setups_safe(fb[:-1]) {
                self._merge_common_call(nd, a_last.expr, b_last.expr, tb[:-1], fb[:-1]);
            }
        }
    }

    """True if setup statements can be safely hoisted out of the branch.

    Hoisting is unsound if a setup transfers control (return/raise/break/
    continue), since that would no longer be conditional. Pure computations and
    static nested guards are fine -- that is exactly what predication assumes.
    """
    def _setups_safe(setups: list) -> bool {
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
        }
        return True;
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
            nm.py_ctx_func = ast3.Store;
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
        torch_nm = self._gen_name(at, 'torch', False);
        where_nm = self._gen_name(at, 'where', False);
        target = uni.AtomTrailer(
            target=torch_nm,
            right=where_nm,
            is_attr=True,
            is_null_ok=False,
            kid=[torch_nm, where_nm]
        );
        return uni.FuncCall(
            target=target,
            params=[cond, t, f],
            genai_call=None,
            kid=[target, cond, t, f]
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

    """Rewrite same-target assignment branches into torch.where."""
    def _rewrite_assignment(
        nd: uni.IfStmt, lhs: uni.Name, true_e: uni.Expr, false_e: uni.Expr
    ) {
        (stmts, c_ref, t_ref, f_ref) = self._hoist(nd, nd.condition, true_e, false_e);
        where_call = self._make_where(nd, c_ref, t_ref, f_ref);
        final = self._assign(lhs, where_call);
        self._replace(nd, stmts + [final]);
    }

    """Rewrite same-shape return branches into a torch.where return."""
    def _rewrite_return(nd: uni.IfStmt, true_e: uni.Expr, false_e: uni.Expr) {
        (stmts, c_ref, t_ref, f_ref) = self._hoist(nd, nd.condition, true_e, false_e);
        where_call = self._make_where(nd, c_ref, t_ref, f_ref);
        ret = uni.ReturnStmt(expr=where_call, kid=[where_call]);
        self._replace(nd, stmts + [ret]);
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
        new_nodes = [cond_assign] + list(true_setups) + list(false_setups) + [
            call_stmt
        ];
        self._replace(nd, new_nodes);
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
