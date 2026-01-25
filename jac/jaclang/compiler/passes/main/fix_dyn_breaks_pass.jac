"""Pytorch 2 Graph Break Fix Pass."""
import ast as ast3;
import from typing { TypeVar, cast }
import jaclang.pycore.unitree as uni;
import from jaclang.pycore.constant { Tokens as Tok }
import from jaclang.pycore.passes.uni_pass { UniPass }
glob T = TypeVar('T', bound=ast3.AST);
"""Fix Dynamic Control Flow Breaks Pass - transforms if statements with dynamic control flow breaks into torch.where."""
class FixDynBreaksPass(UniPass) {
    """Generate Name."""
    def gen_name(
        self: FixDynBreaksPass, <>node: uni.UniNode, name: Tok, value: str
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
    """Replace old node with new nodes in parent's body and kid lists."""
    def replace_node(
        self: FixDynBreaksPass,
        new_nodes: (list[uni.UniNode] | uni.UniNode),
        old_node: uni.UniNode,
        attr: str
    ) -> None {
        parent = old_node.parent;
        if (parent is None) {
            return;
        }
        if isinstance(new_nodes, uni.UniNode) {
            new_nodes.parent = parent;
            if hasattr(parent, attr) {
                lst = getattr(parent, attr);
                if (old_node in lst) {
                    idx = lst.index(old_node);
                    lst[idx] = new_nodes;
                }
            }
            if (old_node in parent.kid) {
                idx = parent.kid.index(old_node);
                parent.kid[idx] = new_nodes;
            }
        } else {
            for n in new_nodes {
                n.parent = parent;
            }
            if hasattr(parent, attr) {
                lst = getattr(parent, attr);
                if (old_node in lst) {
                    idx = lst.index(old_node);
                    setattr(parent, attr, lst[:idx] + new_nodes + lst[(idx + 1):]);
                }
            }
            if (old_node in parent.kid) {
                idx = parent.kid.index(old_node);
                parent.kid = parent.kid[:idx] + new_nodes + parent.kid[(idx + 1):];
            }
        }
    }
    """Return the common LHS target if both are simple assignment with same target."""
    def check_same_lhs(
        self: FixDynBreaksPass, assign_a: uni.UniNode, assign_b: uni.UniNode
    ) -> (uni.Name | None) {
        if not (
            isinstance(assign_a, uni.Assignment)
            and isinstance(assign_b, uni.Assignment)
        ) {
            return None;
        }
        (ta, tb) = (assign_a.target[0], assign_b.target[0]);
        if not (isinstance(ta, uni.Name) and isinstance(tb, uni.Name)) {
            return None;
        }
        if (ta.value != tb.value) {
            return None;
        }
        return ta;
    }
    """Check if node is a method call and return (target, method_name, args, kwargs)."""
    def check_method_call(
        self: FixDynBreaksPass, <>node: uni.ExprStmt
    ) -> (tuple | None) {
        if (
            not isinstance(<>node, uni.ExprStmt)
            or not isinstance(<>node.expr, uni.FuncCall)
        ) {
            return None;
        }
        call = <>node.expr;
        if not isinstance(call.target, uni.AtomTrailer) {
            return None;
        }
        method_name = None;
        if (isinstance(call.target.right, uni.Name) or call.target.right?.value) {
            method_name = call.target.right.value;
        }
        if not method_name {
            return None;
        }
        args = [];
        kwargs = {};
        for param in call.params {
            if isinstance(param, uni.KWPair) {
                if param.key {
                    key_name = param.key.value
                    if isinstance(param.key, uni.Name)
                    else str(param.key);
                    kwargs[key_name] = param.value;
                }
            } else {
                args.append(param);
            }
        }
        return (call.target.target, method_name, args, kwargs);
    }
    """
        Always extract expressions into temporary variables for clarity.
        Returns (assignments_list, cond_name, true_name, false_name).
        """
    def extract_expressions(
        self: FixDynBreaksPass,
        <>node: uni.IfStmt,
        condition: uni.Expr,
        true_expr: uni.Expr,
        false_expr: uni.Expr
    ) -> tuple[list[uni.Assignment], uni.Name, uni.Name, uni.Name] {
        assignments = [];
        cond_tmp = self.gen_name(<>node, Tok.NAME, '__cond');
        cond_tmp.py_ctx_func = ast3.Store;
        cond_assign = uni.Assignment(
            target=[cond_tmp],
            value=condition,
            type_tag=None,
            kid=[cond_tmp, condition]
        );
        assignments.append(cond_assign);
        cond_ref = self.gen_name(<>node, Tok.NAME, '__cond');
        true_tmp = self.gen_name(<>node, Tok.NAME, '__true_val');
        true_tmp.py_ctx_func = ast3.Store;
        true_assign = uni.Assignment(
            target=[true_tmp],
            value=true_expr,
            type_tag=None,
            kid=[true_tmp, true_expr]
        );
        assignments.append(true_assign);
        true_ref = self.gen_name(<>node, Tok.NAME, '__true_val');
        false_tmp = self.gen_name(<>node, Tok.NAME, '__false_val');
        false_tmp.py_ctx_func = ast3.Store;
        false_assign = uni.Assignment(
            target=[false_tmp],
            value=false_expr,
            type_tag=None,
            kid=[false_tmp, false_expr]
        );
        assignments.append(false_assign);
        false_ref = self.gen_name(<>node, Tok.NAME, '__false_val');
        return (assignments, cond_ref, true_ref, false_ref);
    }

    """Exit if statement and transform if it has graph breaks."""
    def exit_if_stmt(self: FixDynBreaksPass, <>node: uni.IfStmt) -> None {
        has_dyn_cf_break = (
            getattr(<>node, 'has_break_dyn_cf', False)
            or getattr(<>node, 'has_dynamo_graph_break', False)
        );
        if not has_dyn_cf_break {
            return;
        }
        if (not <>node.body or (len(<>node.body) != 1)) {
            return;
        }
        if (
            not <>node.else_body
            or not <>node.else_body.body
            or (len(<>node.else_body.body) != 1)
        ) {
            return;
        }
        a0 = <>node.body[0];
        b0 = <>node.else_body.body[0];
        new_node: (uni.UniNode | None) = None;
        if (isinstance(a0, uni.Assignment) and isinstance(b0, uni.Assignment)) {
            lhs = self.check_same_lhs(a0, b0);
            if (lhs is not None) {
                (temp_assigns, cond_ref, true_ref, false_ref) = self.extract_expressions(
                    <>node,
                    <>node.condition,
                    cast(uni.Expr, a0.value),
                    cast(uni.Expr, b0.value)
                );
                func_name = self.gen_name(<>node, Tok.NAME, 'torch');
                attr_name = self.gen_name(<>node, Tok.NAME, 'where');
                target = uni.AtomTrailer(
                    target=func_name,
                    right=attr_name,
                    is_attr=True,
                    is_null_ok=False,
                    kid=[func_name, attr_name]
                );
                call = uni.FuncCall(
                    target=target,
                    params=[cond_ref, true_ref, false_ref],
                    genai_call=None,
                    kid=[target, cond_ref, true_ref, false_ref]
                );
                result_assign = uni.Assignment(
                    target=[lhs], value=call, type_tag=None, kid=[lhs, call]
                );
                all_stmts = temp_assigns + [result_assign];
                self.replace_node(all_stmts, <>node, 'body');
                return;
            }
        } elif (isinstance(a0, uni.ReturnStmt) and isinstance(b0, uni.ReturnStmt)) {
            (aexpr, bexpr) = (a0.expr, b0.expr);
            if ((aexpr is None) or (bexpr is None)) {
                return;
            }
            (temp_assigns, cond_ref, true_ref, false_ref) = self.extract_expressions(
                <>node, <>node.condition, cast(uni.Expr, aexpr), cast(uni.Expr, bexpr)
            );
            func_name = self.gen_name(<>node, Tok.NAME, 'torch');
            attr_name = self.gen_name(<>node, Tok.NAME, 'where');
            target = uni.AtomTrailer(
                target=func_name,
                right=attr_name,
                is_attr=True,
                is_null_ok=False,
                kid=[func_name, attr_name]
            );
            call = uni.FuncCall(
                target=target,
                params=[cond_ref, true_ref, false_ref],
                genai_call=None,
                kid=[target, cond_ref, true_ref, false_ref]
            );
            return_stmt = uni.ReturnStmt(expr=call, kid=[call]);
            all_stmts = temp_assigns + [return_stmt];
            self.replace_node(all_stmts, <>node, 'body');
            return;
        } elif (isinstance(a0, uni.ExprStmt) and isinstance(b0, uni.ExprStmt)) {
            a_call = self.check_method_call(a0);
            b_call = self.check_method_call(b0);
            if ((a_call is not None) and (b_call is not None)) {
                (a_target, a_method, a_args, a_kwargs) = a_call;
                (b_target, b_method, b_args, b_kwargs) = b_call;
                if (
                    (a_method == b_method)
                    and (<>set(a_kwargs.keys()) == <>set(b_kwargs.keys()))
                ) {
                    if ((len(a_args) >= 2) and (len(b_args) >= 2)) {
                        first_arg_same = False;
                        if (
                            isinstance(a_args[0], (uni.String, uni.MultiString))
                            and isinstance(b_args[0], (uni.String, uni.MultiString))
                        ) {
                            a_str = a_args[0].value
                            if isinstance(a_args[0], uni.String)
                            else a_args[0].strings[0].value;
                            b_str = b_args[0].value
                            if isinstance(b_args[0], uni.String)
                            else b_args[0].strings[0].value;
                            first_arg_same = a_str == b_str;
                        }
                        if first_arg_same {
                            (temp_assigns, cond_ref, true_ref, false_ref) = self.extract_expressions(
                                <>node, <>node.condition, a_args[1], b_args[1]
                            );
                            tmp_name_value = f"__{a_args[0].value
                            if isinstance(a_args[0], uni.String)
                            else 'temp'}";
                            tmp_name = self.gen_name(<>node, Tok.NAME, tmp_name_value);
                            tmp_name.py_ctx_func = ast3.Store;
                            torch_name = self.gen_name(<>node, Tok.NAME, 'torch');
                            where_name = self.gen_name(<>node, Tok.NAME, 'where');
                            where_target = uni.AtomTrailer(
                                target=torch_name,
                                right=where_name,
                                is_attr=True,
                                is_null_ok=False,
                                kid=[torch_name, where_name]
                            );
                            where_call = uni.FuncCall(
                                target=where_target,
                                params=[cond_ref, true_ref, false_ref],
                                genai_call=None,
                                kid=[where_target, cond_ref, true_ref, false_ref]
                            );
                            result_assign = uni.Assignment(
                                target=[tmp_name],
                                value=where_call,
                                type_tag=None,
                                kid=[tmp_name, where_call]
                            );
                            kwargs_nodes = [];
                            for (k, v) in a_kwargs.items() {
                                key_node = self.gen_name(<>node, Tok.NAME, k);
                                kwargs_nodes.append(
                                    uni.KWPair(key_node, v, [key_node, v])
                                );
                            }
                            param_name = self.gen_name(
                                <>node, Tok.NAME, tmp_name_value
                            );
                            method_call_params = (
                                [a_args[0], param_name] + kwargs_nodes
                            );
                            original_call_target = a0.expr.target
                            if isinstance(a0.expr, uni.FuncCall)
                            else None;
                            if original_call_target {
                                method_call = uni.FuncCall(
                                    target=original_call_target,
                                    params=method_call_params,
                                    genai_call=None,
                                    kid=([original_call_target] + method_call_params)
                                );
                                method_node = uni.ExprStmt(
                                    expr=method_call,
                                    in_fstring=False,
                                    kid=[method_call]
                                );
                                all_stmts = (
                                    temp_assigns + [result_assign, method_node]
                                );
                                self.replace_node(all_stmts, <>node, 'body');
                                return;
                            }
                        }
                    }
                }
            }
        }
    }
}

