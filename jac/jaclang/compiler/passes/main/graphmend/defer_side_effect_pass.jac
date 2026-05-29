"""GraphMend: Graph-Epilogue Deferred Side Effects transformation.

Side-effecting calls (print / logger.*) inside a compiled region force a graph
break because they touch the Python runtime mid-graph. This pass defers them:
each tagged call (`side_effect` from GraphBreakDetectPass) is replaced by an
append into a per-function buffer, and the buffer is flushed -- in original
program order -- after the compiled graph has run (paper Figs. 5-6).

  @torch.compile
  def fn(x):                          @torch.compile
      x = torch.relu(x)               def fn(x):
      print("t:", x)          ->          _gm_se_buffer = []
      return torch.sin(x)                 x = torch.relu(x)
                                          _gm_se_buffer.append((print, ("t:", x), {}))
                                          __jac_flush_se_buffer__(_gm_se_buffer)
                                          return torch.sin(x)

list.append is a side effect TorchDynamo tracks and replays, so the appends do
not break the graph; only the trailing flush touches the runtime, after all
compute. Each buffer entry is a (callee, args, kwargs) triple so values are
captured at the original call site. Gated by --graphmend.
"""
import ast as ast3;
import jaclang.jac0core.unitree as uni;
import from jaclang.jac0core.passes { UniPass }
import from jaclang.jac0core.constant { Tokens as Tok }

"""Defer print/logger side effects to a post-graph buffer flush."""
obj DeferSideEffectPass(UniPass) {
    has _enabled: bool = False;

    """Gate the pass on the --graphmend flag."""
    def before_pass {
        self._enabled = getattr(self.prog._compile_options, 'graphmend', False);
        if not self._enabled {
            self.prune_signal = True;
        }
    }

    """Replace a tagged side-effect call with a buffer append."""
    def exit_func_call(nd: uni.FuncCall) -> None {
        if not self._enabled {
            return;
        }
        if (nd?._gm_break_kind != 'side_effect') {
            return;
        }
        parent = nd.parent;
        if not isinstance(parent, uni.ExprStmt) {
            return;
        }
        ability = nd.find_parent_of_type(uni.Ability);
        if ability is None {
            return;
        }
        append_call = self._buffer_append(nd, self._buffer_entry(nd));
        parent.expr = append_call;
        parent.set_kids([append_call]);
        ability._gm_needs_se = True;
    }

    """Flush the buffer before any return inside a buffering function."""
    def exit_return_stmt(nd: uni.ReturnStmt) -> None {
        if not self._enabled {
            return;
        }
        ability = nd.find_parent_of_type(uni.Ability);
        if (ability is None) or (not getattr(ability, '_gm_needs_se', False)) {
            return;
        }
        self._insert_before(nd, self._flush_stmt(nd));
    }

    """Initialize the buffer and add a trailing flush for a buffering function."""
    def exit_ability(nd: uni.Ability) -> None {
        if not self._enabled {
            return;
        }
        if not getattr(nd, '_gm_needs_se', False) {
            return;
        }
        body = nd.body;
        if (not isinstance(body, list)) or (not body) {
            return;
        }
        mod = nd.find_parent_of_type(uni.Module);
        if mod is not None {
            mod._gm_needs_se_flush = True;
        }
        self._insert_front(nd, self._buffer_init(nd));
        # If control can fall off the end, flush there too (returns are handled
        # in exit_return_stmt; flush_se_buffer clears, so double-flush is safe).
        tail = nd.body[-1];
        if not isinstance(tail, uni.ReturnStmt) {
            self._append_stmt(nd, self._flush_stmt(nd));
        }
    }

    # ---- node construction helpers ----
    """Mint a synthetic Name anchored at `at`, with Store context if `store`."""
    def _gen_name(at: uni.UniNode, value: str, store: bool) -> uni.Name {
        nm = uni.Name.gen_stub_from_node(at, value);
        if store {
            nm.py_ctx_func = ast3.Store;
        }
        return nm;
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

    """Split call params into positional args and KWPair list."""
    def _split_params(params: any) -> tuple {
        pos = [];
        kws = [];
        for p in (params or []) {
            if isinstance(p, uni.KWPair) {
                kws.append(p);
            } else {
                pos.append(p);
            }
        }
        return (pos, kws);
    }

    """Build a tuple literal anchored at `at`."""
    def _tuple(at: uni.UniNode, values: list) -> uni.TupleVal {
        kid = values or [at.gen_token(Tok.LPAREN, '('), at.gen_token(Tok.RPAREN, ')')];
        return uni.TupleVal(values=values, kid=kid);
    }

    """Build a dict literal from a list of KWPair nodes (keys become strings)."""
    def _dict_from_kws(at: uni.UniNode, kws: list) -> uni.DictVal {
        pairs = [];
        for kw in kws {
            key = self._str(at, kw.key.value);
            pairs.append(uni.KVPair(key=key, value=kw.value, kid=[key, kw.value]));
        }
        kid = pairs or [at.gen_token(Tok.LBRACE, '{'), at.gen_token(Tok.RBRACE, '}')];
        return uni.DictVal(kv_pairs=pairs, kid=kid);
    }

    """Build the `(callee, (args...), {kwargs})` buffer entry for a call."""
    def _buffer_entry(call: uni.FuncCall) -> uni.TupleVal {
        (pos, kws) = self._split_params(call.params);
        args_tuple = self._tuple(call, pos);
        kwargs_dict = self._dict_from_kws(call, kws);
        items = [call.target, args_tuple, kwargs_dict];
        return uni.TupleVal(values=items, kid=items);
    }

    """Build `_gm_se_buffer.append(ent)`."""
    def _buffer_append(at: uni.UniNode, ent: uni.Expr) -> uni.FuncCall {
        buf = self._gen_name(at, '_gm_se_buffer', False);
        attr = self._gen_name(at, 'append', False);
        target = uni.AtomTrailer(
            target=buf, right=attr, is_attr=True, is_null_ok=False, kid=[buf, attr]
        );
        return uni.FuncCall(
            target=target, params=[ent], genai_call=None, kid=[target, ent]
        );
    }

    """Build `_gm_se_buffer = []`."""
    def _buffer_init(at: uni.UniNode) -> uni.Assignment {
        buf = self._gen_name(at, '_gm_se_buffer', True);
        empty = uni.ListVal(
            values=[],
            kid=[at.gen_token(Tok.LSQUARE, '['), at.gen_token(Tok.RSQUARE, ']')]
        );
        return uni.Assignment(
            target=[buf], value=empty, type_tag=None, mutable=True, kid=[buf, empty]
        );
    }

    """Build `__jac_flush_se_buffer__(_gm_se_buffer)` as a statement."""
    def _flush_stmt(at: uni.UniNode) -> uni.ExprStmt {
        helper = self._gen_name(at, '__jac_flush_se_buffer__', False);
        buf = self._gen_name(at, '_gm_se_buffer', False);
        call = uni.FuncCall(
            target=helper, params=[buf], genai_call=None, kid=[helper, buf]
        );
        return uni.ExprStmt(expr=call, in_fstring=False, kid=[call]);
    }

    # ---- body splicing helpers ----
    """Insert `new_stmt` at the front of an ability's body."""
    def _insert_front(ability: uni.Ability, new_stmt: uni.UniNode) {
        body = ability.body;
        first = body[0];
        ability.body = [new_stmt] + list(body);
        if first in ability.kid {
            j = ability.kid.index(first);
            ability.set_kids(
                list(ability.kid[:j]) + [new_stmt] + list(ability.kid[j:])
            );
        }
        new_stmt.parent = ability;
    }

    """Append `new_stmt` to the end of an ability's body."""
    def _append_stmt(ability: uni.Ability, new_stmt: uni.UniNode) {
        body = ability.body;
        last = body[-1];
        ability.body = list(body) + [new_stmt];
        if last in ability.kid {
            j = ability.kid.index(last);
            ability.set_kids(
                list(ability.kid[:(j + 1)]) + [new_stmt] + list(ability.kid[(j + 1):])
            );
        }
        new_stmt.parent = ability;
    }

    """Insert `new_stmt` immediately before `stmt` in its parent body and kid."""
    def _insert_before(stmt: uni.UniNode, new_stmt: uni.UniNode) -> None {
        parent = stmt.parent;
        if parent is None {
            return;
        }
        body = parent?.body;
        if isinstance(body, list) and (stmt in body) {
            i = body.index(stmt);
            parent.body = body[:i] + [new_stmt] + body[i:];
        }
        if stmt in parent.kid {
            j = parent.kid.index(stmt);
            parent.set_kids(list(parent.kid[:j]) + [new_stmt] + list(parent.kid[j:]));
        }
        new_stmt.parent = parent;
    }
}
