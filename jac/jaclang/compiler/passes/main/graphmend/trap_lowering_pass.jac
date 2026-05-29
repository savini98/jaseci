"""GraphMend: Predicated Trap Lowering transformation.

Validation guards of the form `if not <tensor-check>: raise Exc(msg)` (tagged
`val_guard` by GraphBreakDetectPass) raise on a data-dependent boolean, which
breaks tracing. Unlike data-dependent control flow they produce no tensor output,
so torch.where does not apply; instead we lower them to torch._assert_async, a
graph-native assertion that raises at runtime without breaking the graph
(paper Figs. 7-8).

  if not torch.equal(a, b): raise ValueError(msg)
      ->  __jac_tensor_eq_assert__(a, b, msg)     # shape/dtype precheck + (a==b).all()
  if not x.all():           raise ValueError(msg)
      ->  torch._assert_async(x.all(), msg)

torch.equal is special-cased because it returns a Python bool; the runtime helper
rebuilds it as a tensor op with a static shape/dtype guard (see tensor_eq_assert).

Limitation (documented): torch._assert_async raises a RuntimeError rather than the
original exception type. Preserving the exact exception type via a hoisted
error-id wrapper (paper) is left as future work. Gated by --graphmend.
"""
import jaclang.jac0core.unitree as uni;
import from jaclang.jac0core.passes { UniPass }

# FuncCall trailing attributes that already return a tensor boolean, so the
# call can be asserted directly without a Python-bool conversion.
glob _TENSOR_BOOL_OPS: set[str] = {'all', 'any', 'allclose'};

"""Lower data-dependent validation guards to torch._assert_async."""
obj TrapLoweringPass(UniPass) {
    has _enabled: bool = False;

    """Gate the pass on the --graphmend flag."""
    def before_pass {
        self._enabled = getattr(self.prog._compile_options, 'graphmend', False);
        if not self._enabled {
            self.prune_signal = True;
        }
    }

    """Lower a tagged `if not <check>: raise` guard on exit."""
    def exit_if_stmt(nd: uni.IfStmt) -> None {
        if not self._enabled {
            return;
        }
        if (nd?._gm_break_kind != 'val_guard') {
            return;
        }
        if (not nd.body) or (len(nd.body) != 1) {
            return;
        }
        rs = nd.body[0];
        if not isinstance(rs, uni.RaiseStmt) {
            return;
        }
        # Only `if not <X>: raise` is safe to assert as `<X>` must hold.
        cond = nd.condition;
        if not isinstance(cond, uni.UnaryExpr) {
            return;
        }
        if (cond.op is None) or (cond.op.value != 'not') {
            return;
        }
        x = cond.operand;
        msg = self._raise_message(rs, nd);
        eq = self._torch_equal_args(x);
        if eq is not None {
            (a, b) = eq;
            mod = nd.find_parent_of_type(uni.Module);
            if mod is not None {
                mod._gm_needs_eq_assert = True;
            }
            self._replace(nd, [self._eq_assert_stmt(nd, a, b, msg)]);
        } elif self._returns_tensor_bool(x) {
            self._replace(nd, [self._assert_async_stmt(nd, x, msg)]);
        }
    }

    # ---- pattern matching ----
    """Return (a, b) if `x` is torch.equal(a, b), else None."""
    def _torch_equal_args(x: uni.Expr) -> (tuple | None) {
        if not isinstance(x, uni.FuncCall) {
            return None;
        }
        target = x.target;
        if not (
            isinstance(target, uni.AtomTrailer)
            and isinstance(target.right, uni.Name)
            and (target.right.value == 'equal')
            and isinstance(target.target, uni.Name)
            and (target.target.value == 'torch')
        ) {
            return None;
        }
        pos = [
            p
            for p in (x.params or [])
            if not isinstance(p, uni.KWPair)
        ];
        if len(pos) < 2 {
            return None;
        }
        return (pos[0], pos[1]);
    }

    """True if `x` is a call returning a tensor boolean (e.g. x.all())."""
    def _returns_tensor_bool(x: uni.Expr) -> bool {
        return (
            isinstance(x, uni.FuncCall)
            and isinstance(x.target, uni.AtomTrailer)
            and isinstance(x.target.right, uni.Name)
            and (x.target.right.value in _TENSOR_BOOL_OPS)
        );
    }

    """Extract the message expression from a raise, or synthesize an empty str."""
    def _raise_message(rs: uni.RaiseStmt, at: uni.UniNode) -> uni.Expr {
        cause = rs.cause;
        if isinstance(cause, uni.FuncCall) {
            for p in (cause.params or []) {
                if not isinstance(p, uni.KWPair) {
                    return p;
                }
            }
        }
        return self._str(at, '');
    }

    # ---- node construction ----
    """Mint a synthetic Name anchored at `at`."""
    def _gen_name(at: uni.UniNode, value: str) -> uni.Name {
        return uni.Name.gen_stub_from_node(at, value);
    }

    """Mint a synthetic string literal (value carries its quotes)."""
    def _str(at: uni.UniNode, text: str) -> uni.String {
        return uni.String(
            orig_src=at.loc.orig_src,
            name='STRING',
            value=f'"{text}"',
            line=at.loc.first_line,
            end_line=at.loc.last_line,
            col_start=at.loc.col_start,
            col_end=at.loc.col_end,
            pos_start=at.loc.pos_start,
            pos_end=at.loc.pos_end
        );
    }

    """Wrap a call expression in an expression statement."""
    def _call_stmt(call: uni.FuncCall) -> uni.ExprStmt {
        return uni.ExprStmt(expr=call, in_fstring=False, kid=[call]);
    }

    """Build `__jac_tensor_eq_assert__(a, b, msg)`."""
    def _eq_assert_stmt(
        at: uni.UniNode, a: uni.Expr, b: uni.Expr, msg: uni.Expr
    ) -> uni.ExprStmt {
        helper = self._gen_name(at, '__jac_tensor_eq_assert__');
        params = [a, b, msg];
        call = uni.FuncCall(
            target=helper, params=params, genai_call=None, kid=[helper] + params
        );
        return self._call_stmt(call);
    }

    """Build `torch._assert_async(x, msg)`."""
    def _assert_async_stmt(
        at: uni.UniNode, x: uni.Expr, msg: uni.Expr
    ) -> uni.ExprStmt {
        torch_nm = self._gen_name(at, 'torch');
        assert_nm = self._gen_name(at, '_assert_async');
        target = uni.AtomTrailer(
            target=torch_nm,
            right=assert_nm,
            is_attr=True,
            is_null_ok=False,
            kid=[torch_nm, assert_nm]
        );
        params = [x, msg];
        call = uni.FuncCall(
            target=target, params=params, genai_call=None, kid=[target] + params
        );
        return self._call_stmt(call);
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
            parent.set_kids(
                list(parent.kid[:j]) + new_nodes + list(parent.kid[(j + 1):])
            );
        }
        for child in new_nodes {
            child.parent = parent;
        }
    }
}
