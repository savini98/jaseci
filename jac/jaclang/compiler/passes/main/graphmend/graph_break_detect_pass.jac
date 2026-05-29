"""GraphMend: detect PyTorch 2 graph-break locations and tag them.

Implements the paper's Dynamo Entry Point Analysis + Graph Break Type Analysis
(Algorithm 1). It walks torch.compile-decorated functions/classes and tags AST
nodes with a `_gm_break_kind` marker consumed by the transformation passes:

  - "dyn_ctrl_fl" : data-dependent if/else      -> PredicateCtrlFlowPass (torch.where)
  - "val_guard"   : data-dependent guard `raise` -> TrapLoweringPass (_assert_async)
  - "side_effect" : print / logger call          -> DeferSideEffectPass (buffered flush)

Detection is generalized (not a fixed op list): a condition is data-dependent if
it uses any dynamic torch attribute (see TorchAttrTable) directly, or if a
use-def trace over the symbol table shows it derives from a tensor input or a
dynamic torch op.

Gated by --graphmend: when the flag is off, or the module has no torch.compile
entry point, the pass prunes immediately and does nothing.
"""
import jaclang.jac0core.unitree as uni;
import from jaclang.jac0core.passes { UniPass }
import from jaclang.compiler.passes.main.graphmend.torch_attr_table {
    is_dynamic_torch_attr,
    is_side_effect_name
}

"""Tag graph-break locations under torch.compile entry points."""
obj GraphBreakDetectPass(UniPass) {
    has _enabled: bool = False;

    """Gate the pass on the --graphmend flag and presence of an entry point."""
    def before_pass -> None {
        self._enabled = getattr(self.prog._compile_options, 'graphmend', False);
        if not self._enabled {
            self.prune_signal = True;
            return;
        }
        if not self._module_has_torch_compile() {
            self._enabled = False;
            self.prune_signal = True;
        }
    }

    """Dynamo Entry Point Analysis: does any def/class carry @torch.compile?"""
    def _module_has_torch_compile -> bool {
        for ab in UniPass.get_all_sub_nodes(self.ir_in, uni.Ability, True) {
            if self._has_torch_compile_decorator(ab) {
                return True;
            }
        }
        for arch in UniPass.get_all_sub_nodes(self.ir_in, uni.Archetype, True) {
            if self._has_torch_compile_decorator(arch) {
                return True;
            }
        }
        return False;
    }

    """True if `nd` is decorated with torch.compile (with or without args)."""
    def _has_torch_compile_decorator(nd: uni.UniNode) -> bool {
        decs = nd?.decorators;
        for dec in (decs or []) {
            # unparse() pads tokens with spaces (e.g. "torch.compile ( )"),
            # so normalize before matching.
            txt = dec.unparse().replace(' ', '');
            if (txt == 'torch.compile') or txt.startswith('torch.compile(') {
                return True;
            }
        }
        return False;
    }

    """True if `nd` lives inside a torch.compile-decorated def or class."""
    def _in_torch_compiled_scope(nd: uni.UniNode) -> bool {
        cur = nd.parent;
        while cur {
            if isinstance(cur, (uni.Ability, uni.Archetype)) {
                if self._has_torch_compile_decorator(cur) {
                    return True;
                }
            }
            cur = cur.parent;
        }
        return False;
    }

    """Tag data-dependent if/else and validation-guard conditionals."""
    def enter_if_stmt(nd: uni.IfStmt) -> None {
        if not self._enabled {
            return;
        }
        # elif heads are part of the parent chain; only tag leading `if`.
        if isinstance(nd, uni.ElseIf) {
            return;
        }
        if not self._in_torch_compiled_scope(nd) {
            return;
        }
        if not self._condition_is_data_dependent(nd.condition) {
            return;
        }
        if self._body_is_only_raise(nd.body) {
            nd._gm_break_kind = 'val_guard';
        } else {
            nd._gm_break_kind = 'dyn_ctrl_fl';
        }
    }

    """Tag print / logger side-effect calls."""
    def enter_func_call(nd: uni.FuncCall) -> None {
        if not self._enabled {
            return;
        }
        if not self._in_torch_compiled_scope(nd) {
            return;
        }
        name = self._call_name(nd);
        if (name is not None) and is_side_effect_name(name) {
            if self._is_user_defined(nd, name) {
                return;
            }
            nd._gm_break_kind = 'side_effect';
        }
    }

    """Get the callee name of a FuncCall (bare name or trailing attribute)."""
    def _call_name(nd: uni.FuncCall) -> (str | None) {
        target = nd.target;
        if isinstance(target, uni.Name) {
            return target.value;
        }
        if (
            isinstance(target, uni.AtomTrailer) and isinstance(target.right, uni.Name)
        ) {
            return target.right.value;
        }
        return None;
    }

    """True if `name` resolves to a user-defined symbol (shadowed builtin)."""
    def _is_user_defined(nd: uni.FuncCall, name: str) -> bool {
        st = nd.sym_tab;
        if not st {
            return False;
        }
        sym = st.lookup(name, deep=True);
        return bool(sym and sym.defn);
    }

    """True if a branch body is exactly one `raise` statement."""
    def _body_is_only_raise(body: list) -> bool {
        if (not body) or (len(body) != 1) {
            return False;
        }
        return isinstance(body[0], uni.RaiseStmt);
    }

    """Algorithm 1 condition check: dynamic torch attr or tensor use-def."""
    def _condition_is_data_dependent(cond: uni.Expr) -> bool {
        if self._expr_uses_dynamic_attr(cond) {
            return True;
        }
        st = cond.sym_tab;
        if not st {
            return False;
        }
        visited: set[str] = set();
        for nm in cond.get_all_sub_nodes(uni.Name) {
            sym = st.lookup(nm.value, deep=True);
            if sym and self._trace_symbol(sym, visited, 0) {
                return True;
            }
        }
        return False;
    }

    """True if any attribute trailer in `expr` names a dynamic torch op."""
    def _expr_uses_dynamic_attr(expr: uni.Expr) -> bool {
        for tr in expr.get_all_sub_nodes(uni.AtomTrailer) {
            if tr.is_attr and isinstance(tr.right, uni.Name) {
                if is_dynamic_torch_attr(tr.right.value) {
                    return True;
                }
            }
        }
        return False;
    }

    """Trace a symbol's definition chain for tensor/parameter dependence."""
    def _trace_symbol(sym: uni.Symbol, visited: set[str], depth: int) -> bool {
        if (depth > 10) or (sym.sym_name in visited) {
            return False;
        }
        visited.add(sym.sym_name);
        if self._symbol_is_parameter(sym) {
            return True;
        }
        assign = self._defining_assignment(sym);
        if (assign is None) or (assign.value is None) {
            return False;
        }
        if self._expr_uses_dynamic_attr(assign.value) {
            return True;
        }
        st = assign.sym_tab;
        if not st {
            return False;
        }
        for nm in assign.value.get_all_sub_nodes(uni.Name) {
            dep = st.lookup(nm.value, deep=True);
            if dep and (dep.sym_name not in visited) {
                if self._trace_symbol(dep, visited, depth + 1) {
                    return True;
                }
            }
        }
        return False;
    }

    """True if the symbol is a function parameter (a model input)."""
    def _symbol_is_parameter(sym: uni.Symbol) -> bool {
        if not sym.defn {
            return False;
        }
        cur = sym.defn[0].parent;
        while cur {
            if isinstance(cur, uni.ParamVar) {
                return True;
            }
            if isinstance(cur, uni.Ability) {
                return False;
            }
            cur = cur.parent;
        }
        return False;
    }

    """Return the Assignment that defines `sym`, if any."""
    def _defining_assignment(sym: uni.Symbol) -> (uni.Assignment | None) {
        if not sym.defn {
            return None;
        }
        cur = sym.defn[-1].parent;
        while cur and (not isinstance(cur, uni.Assignment)) {
            cur = cur.parent;
        }
        if isinstance(cur, uni.Assignment) {
            return cur;
        }
        return None;
    }
}
