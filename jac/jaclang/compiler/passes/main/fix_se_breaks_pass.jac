"""Fix Side Effect Breaks Pass - Transform side-effecting calls to be deferred."""
import ast as ast3;
import from collections.abc { Sequence }
import from copy { deepcopy }
import from typing { cast }
import jaclang.pycore.unitree as uni;
import from jaclang.pycore.constant { Tokens as Tok }
import from jaclang.pycore.passes.uni_pass { UniPass }
"""Fix Side Effect Breaks Pass - buffers side-effecting operations for deferred execution."""
class FixSEBreaksPass(UniPass) {
    """Initialize pass state."""
    def before_pass(self: FixSEBreaksPass) -> None {
        self.needs_se_buffer = False;
        return super.before_pass();
    }

    """Generate a Name node."""
    def gen_name(
        self: FixSEBreaksPass, <>node: uni.UniNode, name: Tok, value: str
    ) -> uni.Name {
        return uni.Name(
            name=name,
            value=value,
            orig_src=<>node.loc.orig_src,
            col_start=<>node.loc.col_start,
            col_end=0,
            line=<>node.loc.first_line,
            end_line=<>node.loc.last_line,
            pos_start=0,
            pos_end=0
        );
    }

    """Replace side-effect call with a call to append to the buffer."""
    def _replace_side_effect_call(
        self: FixSEBreaksPass, <>node: uni.FuncCall
    ) -> uni.FuncCall {
        if isinstance(<>node.target, uni.Name) {
            func_name_str = <>node.target.value;
        } elif isinstance(<>node.target, uni.AtomTrailer) {
            if isinstance(<>node.target.right, uni.Name) {
                func_name_str = <>node.target.right.value;
            } else {
                func_name_str = 'unknown_call';
            }
        } else {
            func_name_str = 'unknown_call';
        }
        func_str = self.gen_name(<>node, Tok.STRING, f'"{func_name_str}"');
        params = deepcopy(<>node.params) if <>node.params else [];
        tuple_params = uni.TupleVal(
            values=cast(Sequence[uni.Expr], params), kid=params
        );
        lpr = self.gen_name(<>node, Tok.LPAREN, '(');
        rpr = self.gen_name(<>node, Tok.RPAREN, ')');
        dict_val = uni.DictVal(kv_pairs=[], kid=[lpr, rpr]);
        tuple_items = [func_str, tuple_params, dict_val];
        buffer_entry = uni.TupleVal(
            values=cast(Sequence[uni.Expr], tuple_items), kid=tuple_items
        );
        args = [buffer_entry];
        buffer_name = self.gen_name(<>node, Tok.NAME, '_se_buffer');
        append_attr = self.gen_name(<>node, Tok.NAME, 'append');
        func_target = uni.AtomTrailer(
            target=buffer_name,
            right=append_attr,
            is_attr=True,
            is_null_ok=False,
            kid=[buffer_name, append_attr]
        );
        return uni.FuncCall(
            target=func_target,
            params=args,
            genai_call=None,
            kid=([func_target] + args)
        );
    }

    """Exit function call - replace if marked with side effect break by CatchBreaksPass."""
    def exit_func_call(self: FixSEBreaksPass, <>node: uni.FuncCall) -> None {
        if not getattr(<>node, 'has_break_se', False) {
            return;
        }
        ability_node = <>node.find_parent_of_type(uni.Ability);
        if (ability_node is not None) {
            ability_node.needs_se_buffer = True;
        }
        new_call = self._replace_side_effect_call(<>node);
        if isinstance(<>node.parent, uni.ExprStmt) {
            <>node.parent.expr = new_call;
            new_call.parent = <>node.parent;
            if (<>node.parent?.kid and (<>node in <>node.parent.kid)) {
                idx = <>node.parent.kid.index(<>node);
                <>node.parent.kid[idx] = new_call;
            }
        }
    }

    """Exit ability - wrap if it needs side effect buffering."""
    def exit_ability(self: FixSEBreaksPass, <>node: uni.Ability) -> None {
        if not getattr(<>node, 'needs_se_buffer', False) {
            return;
        }
        self.needs_se_buffer = True;
        if isinstance(<>node.body, <>list) {
            body = <>node.body;
            body_parent = <>node;
        } elif isinstance(<>node.body, uni.ImplDef) {
            if isinstance(<>node.body.body, <>list) {
                body = <>node.body.body;
                body_parent = <>node.body;
            } else {
                return;
            }
        } else {
            return;
        }
        buffer_name = self.gen_name(<>node, Tok.NAME, '_se_buffer');
        buffer_name.py_ctx_func = ast3.Store;
        lbrak = <>node.gen_token(Tok.LBRACE, '[');
        rbrak = <>node.gen_token(Tok.RBRACE, ']');
        empty_list = uni.ListVal(values=[], kid=[lbrak, rbrak]);
        buffer_init = uni.Assignment(
            target=[buffer_name],
            value=empty_list,
            type_tag=None,
            kid=[buffer_name, empty_list]
        );
        body.insert(0, buffer_init);
        buffer_init.parent = body_parent;
        if ((body_parent == <>node) and <>node?.kid) {
            lbrace_idx = None;
            for (i, k) in enumerate(<>node.kid) {
                if (isinstance(k, uni.Token) and (k.value == '{')) {
                    lbrace_idx = i;
                    break;
                }
            }
            if (lbrace_idx is not None) {
                <>node.kid.insert((lbrace_idx + 1), buffer_init);
            }
        }
        self._add_buffer_flush(body, body_parent, <>node);
    }

    """Add buffer flush logic before return statements."""
    def _add_buffer_flush(
        self: FixSEBreaksPass,
        body: list[uni.CodeBlockStmt],
        body_parent: uni.UniNode,
        ability_node: uni.Ability
    ) -> None {
        return_stmts = self._find_return_statements_with_context(body, body_parent);
        if not return_stmts {
            flush_stmt = self._create_flush_statement(ability_node);
            body.append(flush_stmt);
            flush_stmt.parent = body_parent;
            if ((body_parent == ability_node) and ability_node?.kid) {
                rbrace_idx = None;
                for (i, k) in enumerate(ability_node.kid) {
                    if (isinstance(k, uni.Token) and (k.value == '}')) {
                        rbrace_idx = i;
                        break;
                    }
                }
                if (rbrace_idx is not None) {
                    ability_node.kid.insert(rbrace_idx, flush_stmt);
                }
            }
        } else {
            for (ret_stmt, parent_body, parent_node) in return_stmts {
                flush_stmt = self._create_flush_statement(ability_node);
                if (ret_stmt in parent_body) {
                    idx = parent_body.index(ret_stmt);
                    parent_body.insert(idx, flush_stmt);
                    flush_stmt.parent = parent_node;
                    if (parent_node?.kid and (ret_stmt in parent_node.kid)) {
                        kid_idx = parent_node.kid.index(ret_stmt);
                        parent_node.kid.insert(kid_idx, flush_stmt);
                    }
                }
            }
        }
    }

    """Recursively find all return statements with their parent context.

        Returns list of (return_stmt, parent_body_list, parent_node) tuples.
        """
    def _find_return_statements_with_context(
        self: FixSEBreaksPass, stmts: list[uni.CodeBlockStmt], parent: uni.UniNode
    ) -> list[tuple[(uni.ReturnStmt, list[uni.CodeBlockStmt], uni.UniNode)]] {
        returns = [];
        for stmt in stmts {
            if isinstance(stmt, uni.ReturnStmt) {
                returns.append((stmt, stmts, parent));
            } elif isinstance(stmt, uni.IfStmt) {
                returns.extend(
                    self._find_return_statements_with_context(stmt.body, stmt)
                );
                if stmt.else_body {
                    if stmt.else_body?.body {
                        returns.extend(
                            self._find_return_statements_with_context(
                                stmt.else_body.body, stmt.else_body
                            )
                        );
                    }
                }
            } elif (stmt?.body and isinstance(stmt.body, <>list)) {
                returns.extend(
                    self._find_return_statements_with_context(stmt.body, stmt)
                );
            }
        }
        return returns;
    }

    """Create a statement to flush the buffer using a runtime helper.

        Creates:
            __jac_flush_se_buffer__(_se_buffer)

        The runtime helper will execute all buffered calls.
        """
    def _create_flush_statement(
        self: FixSEBreaksPass, <>node: uni.Ability
    ) -> uni.CodeBlockStmt {
        helper_name = self.gen_name(<>node, Tok.NAME, '__jac_flush_se_buffer__');
        buffer_ref = self.gen_name(<>node, Tok.NAME, '_se_buffer');
        flush_call = uni.FuncCall(
            target=helper_name,
            params=[buffer_ref],
            genai_call=None,
            kid=[helper_name, buffer_ref]
        );
        flush_stmt = uni.ExprStmt(expr=flush_call, in_fstring=False, kid=[flush_call]);
        flush_call.parent = flush_stmt;
        return flush_stmt;
    }

    """Exit module - inject runtime helper if needed."""
    def exit_module(self: FixSEBreaksPass, <>node: uni.Module) -> None {
        if not self.needs_se_buffer {
            return;
        }
        ;
    }
}
