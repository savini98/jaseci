"""Fix Side Effect Breaks Pass - Transform side-effecting calls to be deferred."""

import ast as ast3
from copy import deepcopy
from typing import Sequence, cast

import jaclang.pycore.unitree as uni
from jaclang.pycore.constant import Tokens as Tok
from jaclang.pycore.passes.uni_pass import UniPass


class FixSEBreaksPass(UniPass):
    """Fix Side Effect Breaks Pass - buffers side-effecting operations for deferred execution."""

    # List of function names that cause side effects and should be hoisted
    _SIDE_EFFECT_CALLS = {"print", "input"}
    
    # Logging patterns that should be hoisted
    _LOGGING_PATTERNS = ["log", "logger", "logging", "warn", "error", "info", "debug"]

    def before_pass(self) -> None:
        """Initialize pass state."""
        self.needs_se_buffer = False
        return super().before_pass()

    def gen_name(self, node: uni.UniNode, name: Tok, value: str) -> uni.Name:
        """Generate a Name node."""
        return uni.Name(
            name=name,
            value=value,
            orig_src=node.loc.orig_src,
            col_start=node.loc.col_start,
            col_end=0,
            line=node.loc.first_line,
            end_line=node.loc.last_line,
            pos_start=0,
            pos_end=0,
        )

    def _is_side_effect_call(self, node: uni.FuncCall) -> bool:
        """Check if a function call has side effects and should be buffered."""
        if isinstance(node.target, uni.Name):
            func_name = node.target.value
            
            # Check if it's a builtin side-effecting function
            # Verify it's not user-redefined by checking symbol table
            if func_name in self._SIDE_EFFECT_CALLS:
                if hasattr(node, 'sym_tab') and node.sym_tab:
                    symbol = node.sym_tab.lookup(name=func_name, deep=True)
                    if symbol and symbol.defn:
                        # User has redefined this name, skip it
                        return False
                return True
            
            # Check for logging patterns
            for pattern in self._LOGGING_PATTERNS:
                if pattern in func_name.lower():
                    return True
                    
        elif isinstance(node.target, uni.AtomTrailer):
            # Handle method calls like logger.info(), logging.debug()
            parts = []
            current = node.target
            
            while isinstance(current, uni.AtomTrailer):
                if hasattr(current, "right") and isinstance(current.right, uni.Name):
                    parts.append(current.right.value)
                current = current.target
                if isinstance(current, uni.Name):
                    parts.append(current.value)
                    break
            
            # Check if any part matches logging patterns
            for part in parts:
                for pattern in self._LOGGING_PATTERNS:
                    if pattern in part.lower():
                        return True
                        
        return False

    def _replace_side_effect_call(self, node: uni.FuncCall) -> uni.FuncCall:
        """Replace side-effect call with a call to append to the buffer."""
        # Get the function name as a string
        if isinstance(node.target, uni.Name):
            func_name_str = node.target.value
        elif isinstance(node.target, uni.AtomTrailer):
            # For method calls, try to get a meaningful name
            if isinstance(node.target.right, uni.Name):
                func_name_str = node.target.right.value
            else:
                func_name_str = "unknown_call"
        else:
            func_name_str = "unknown_call"
        
        # Create string literal for function name
        func_str = self.gen_name(node, Tok.STRING, f'"{func_name_str}"')
        
        # Create tuple of arguments
        params = deepcopy(node.params) if node.params else []
        tuple_params = uni.TupleVal(
            values=cast(Sequence[uni.Expr], params),
            kid=params
        )
        
        # Create empty dict for kwargs (for now, we don't handle kwargs)
        lpr = self.gen_name(node, Tok.LPAREN, "(")
        rpr = self.gen_name(node, Tok.RPAREN, ")")
        dict_val = uni.DictVal(kv_pairs=[], kid=[lpr, rpr])
        
        # Create tuple: (func_name, args, kwargs)
        tuple_items = [func_str, tuple_params, dict_val]
        buffer_entry = uni.TupleVal(
            values=cast(Sequence[uni.Expr], tuple_items),
            kid=tuple_items
        )
        
        # Create: _se_buffer.append((func_name, args, kwargs))
        args = [buffer_entry]
        
        buffer_name = self.gen_name(node, Tok.NAME, "_se_buffer")
        append_attr = self.gen_name(node, Tok.NAME, "append")
        
        func_target = uni.AtomTrailer(
            target=buffer_name,
            right=append_attr,
            is_attr=True,
            is_null_ok=False,
            kid=[buffer_name, append_attr],
        )
        
        return uni.FuncCall(
            target=func_target,
            params=args,
            genai_call=None,
            kid=[func_target] + args,
        )
    def exit_func_call(self, node: uni.FuncCall) -> None:
        """Exit function call - mark ability if it contains side effects."""
        # Only process if marked as side effect break by CatchBreaksPass
        if not getattr(node.parent, "has_break_se", False):
            return
            
        if self._is_side_effect_call(node):
            # Find the containing ability
            ability_node = node.find_parent_of_type(uni.Ability)
            if ability_node is not None:
                # Mark this ability as needing side effect buffering
                ability_node.needs_se_buffer = True  # type: ignore[attr-defined]
            
            # Replace the call with buffer append
            new_call = self._replace_side_effect_call(node)
            
            # Update parent references
            if isinstance(node.parent, uni.ExprStmt):
                node.parent.expr = new_call
                new_call.parent = node.parent
                if hasattr(node.parent, "kid") and node in node.parent.kid:
                    idx = node.parent.kid.index(node)
                    node.parent.kid[idx] = new_call

    def exit_ability(self, node: uni.Ability) -> None:
        """Exit ability - wrap if it needs side effect buffering."""
        if not getattr(node, "needs_se_buffer", False):
            return
        
        self.needs_se_buffer = True
        
        # Get the body
        if isinstance(node.body, list):
            body = node.body
            body_parent = node
        elif isinstance(node.body, uni.ImplDef):
            if isinstance(node.body.body, list):
                body = node.body.body
                body_parent = node.body
            else:
                return
        else:
            return
        
        # Simple transformation: just initialize the buffer at the start
        # Create: _se_buffer = []
        buffer_name = self.gen_name(node, Tok.NAME, "_se_buffer")
        buffer_name.py_ctx_func = ast3.Store
        
        # Create empty list literal using tokens generated from existing node
        lbrak = node.gen_token(Tok.LBRACE, "[")  # Use LBRACE with value "["
        rbrak = node.gen_token(Tok.RBRACE, "]")  # Use RBRACE with value "]"
        empty_list = uni.ListVal(values=[], kid=[lbrak, rbrak])
        
        # Create assignment: _se_buffer = []
        buffer_init = uni.Assignment(
            target=[buffer_name],
            value=empty_list,
            type_tag=None,
            kid=[buffer_name, empty_list],
        )
        
        # Insert at the beginning of the function body
        body.insert(0, buffer_init)
        buffer_init.parent = body_parent
        
        # IMPORTANT: Also update the node's kid list!
        # When body is a list, it gets added to kid during normalize()
        # We need to insert into kid as well so PyastGenPass can traverse it
        if body_parent == node and hasattr(node, 'kid'):
            # Find the position after the opening brace token in kid
            # The structure is: [...tokens...] LBRACE [body statements] RBRACE
            lbrace_idx = None
            for i, k in enumerate(node.kid):
                if isinstance(k, uni.Token) and k.value == "{":
                    lbrace_idx = i
                    break
            
            if lbrace_idx is not None:
                # Insert after the LBRACE token
                node.kid.insert(lbrace_idx + 1, buffer_init)
        
        # Add buffer flush logic before return statements
        self._add_buffer_flush(body, body_parent, node)

    def _add_buffer_flush(self, body: list[uni.CodeBlockStmt], body_parent: uni.UniNode, ability_node: uni.Ability) -> None:
        """Add buffer flush logic before return statements."""
        # Find all return statements with their parent context
        return_stmts = self._find_return_statements_with_context(body, body_parent)
        
        if not return_stmts:
            # No explicit return - add flush at end
            flush_stmt = self._create_flush_statement(ability_node)
            body.append(flush_stmt)
            flush_stmt.parent = body_parent
            
            # Also add to kid list if needed
            if body_parent == ability_node and hasattr(ability_node, 'kid'):
                # Find RBRACE and insert before it
                rbrace_idx = None
                for i, k in enumerate(ability_node.kid):
                    if isinstance(k, uni.Token) and k.value == "}":
                        rbrace_idx = i
                        break
                if rbrace_idx is not None:
                    ability_node.kid.insert(rbrace_idx, flush_stmt)
        else:
            # Insert flush before each return
            for ret_stmt, parent_body, parent_node in return_stmts:
                flush_stmt = self._create_flush_statement(ability_node)
                
                # Find position in the parent body
                if ret_stmt in parent_body:
                    idx = parent_body.index(ret_stmt)
                    parent_body.insert(idx, flush_stmt)
                    flush_stmt.parent = parent_node
                    
                    # Also update kid list if the parent node has one
                    if hasattr(parent_node, 'kid') and ret_stmt in parent_node.kid:
                        kid_idx = parent_node.kid.index(ret_stmt)
                        parent_node.kid.insert(kid_idx, flush_stmt)

    def _find_return_statements_with_context(
        self, stmts: list[uni.CodeBlockStmt], parent: uni.UniNode
    ) -> list[tuple[uni.ReturnStmt, list[uni.CodeBlockStmt], uni.UniNode]]:
        """Recursively find all return statements with their parent context.
        
        Returns list of (return_stmt, parent_body_list, parent_node) tuples.
        """
        returns = []
        for stmt in stmts:
            if isinstance(stmt, uni.ReturnStmt):
                returns.append((stmt, stmts, parent))
            elif isinstance(stmt, uni.IfStmt):
                # Check if body
                returns.extend(self._find_return_statements_with_context(stmt.body, stmt))
                # Check else_body (can be ElseStmt or ElseIf)
                if stmt.else_body:
                    if hasattr(stmt.else_body, 'body'):
                        returns.extend(self._find_return_statements_with_context(stmt.else_body.body, stmt.else_body))
            elif hasattr(stmt, 'body') and isinstance(stmt.body, list):
                returns.extend(self._find_return_statements_with_context(stmt.body, stmt))
        return returns


    def _create_flush_statement(self, node: uni.Ability) -> uni.CodeBlockStmt:
        """Create a statement to flush the buffer using a runtime helper.
        
        Creates:
            __jac_flush_se_buffer__(_se_buffer)
        
        The runtime helper will execute all buffered calls.
        """
        # Create the helper function name
        helper_name = self.gen_name(node, Tok.NAME, "__jac_flush_se_buffer__")
        
        # Create buffer reference as argument
        buffer_ref = self.gen_name(node, Tok.NAME, "_se_buffer")
        
        # Create function call: __jac_flush_se_buffer__(_se_buffer)
        flush_call = uni.FuncCall(
            target=helper_name,
            params=[buffer_ref],
            genai_call=None,
            kid=[helper_name, buffer_ref]
        )
        
        # Wrap in ExprStmt
        flush_stmt = uni.ExprStmt(
            expr=flush_call,
            in_fstring=False,
            kid=[flush_call]
        )
        flush_call.parent = flush_stmt
        
        return flush_stmt

    def exit_module(self, node: uni.Module) -> None:
        """Exit module - inject runtime helper if needed."""
        if not self.needs_se_buffer:
            return
        
        # Instead of importing, inject the helper function definition inline
        # This avoids complex import IR node creation
        # The helper will be defined at module level
        
        # For now, we rely on the helper being available in the runtime
        # When PyastGenPass generates Python, it will include the call
        # The actual import can be added later or the function can be inlined
        pass
