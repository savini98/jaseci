"""Pytorch Fix Pass."""

import jaclang.pycore.unitree as uni
from jaclang.pycore.passes.uni_pass import UniPass


class CatchBreaksPass(UniPass):
    """Catch Breaks Pass to handle breaks in loops."""

    def before_pass(self):
        """Before pass setup."""
        self._torch_compiled_abilities: list[uni.Ability] = []
        self._torch_compiled_archetypes: list[uni.Archetype] = []
        self._current_archetype: uni.Archetype | None = None
        return super().before_pass()

    def enter_archetype(self, node: uni.Archetype) -> None:
        """Enter archetype (class)."""
        self._current_archetype = node
        # Check if the class itself is decorated with torch.compile
        if node.decorators:
            for dec in node.decorators:
                dec_txt = dec.unparse().strip() if hasattr(dec, "unparse") else ""
                if dec_txt == "torch.compile" or dec_txt.startswith("torch.compile ("):
                    self._torch_compiled_archetypes.append(node)
                    # print(f"Found @torch.compile on class: {node.name.value}")
                    break

    def exit_archetype(self, node: uni.Archetype) -> None:
        """Exit archetype (class)."""
        if self._current_archetype == node:
            self._current_archetype = None

    def enter_ability(self, node: uni.Ability) -> None:
        """Enter ability (function/method)."""
        should_analyze = False

        # Check if the ability itself is decorated with torch.compile
        if node.decorators:
            for dec in node.decorators:
                dec_txt = dec.unparse().strip() if hasattr(dec, "unparse") else ""
                if dec_txt == "torch.compile" or dec_txt.startswith("torch.compile ("):
                    self._torch_compiled_abilities.append(node)
                    should_analyze = True
                    # print(f"Found @torch.compile on function: {node.name_spec.value if hasattr(node, 'name_spec') else 'unknown'}")
                    break
        
        # Check if the ability is inside a torch.compile decorated class
        if not should_analyze and self._current_archetype in self._torch_compiled_archetypes:
            should_analyze = True
            print(f"Found function inside @torch.compile class: {node.name_spec.value if hasattr(node, 'name_spec') else 'unknown'}")
        
        if should_analyze:
            _ = BreakFinder(node)  # Analysis happens in __init__


class CFGTracer:
    """CFG Tracer to trace control flow graphs."""

    def __init__(self, ability: uni.Ability):
        """Initialize CFGTracer."""
        self.ability = ability
        self.run_analysis(self.ability)

    def run_analysis(self, current_node: uni.UniCFGNode) -> None:
        """Analyze symbols."""
        self.analysis_on_basicblock(current_node)
        if current_node.get_tail().bb_out == []:
            return
        for succ in current_node.get_tail().bb_out:
            self.run_analysis(succ)

    def analysis_on_basicblock(self, current_node: uni.UniCFGNode) -> None:
        """Perform analysis."""
        basicblock = current_node.get_current_bb()
        for stmt in basicblock:
            self.analysis_on_stmt(stmt)

    def analysis_on_stmt(self, stmt: uni.UniCFGNode) -> None:
        """Perform analysis."""
        # To be overridden in subclasses


class BreakFinder(CFGTracer):
    """Break Finder to find break statements."""

    def __init__(self, ability: uni.Ability):
        """Initialize BreakFinder."""
        self.breaks: list = []
        self.graph_break_stmts: list[uni.IfStmt] = []  # Track statements with dynamic control flow breaks
        self.side_effect_stmts: list[uni.UniNode] = []  # Track statements with side effects
        self.analyzed_stmts: set[int] = set()  # Track analyzed statements by id to avoid duplicates
        super().__init__(ability)
        # Report summary after analysis
        self.report_graph_breaks()

    def report_graph_breaks(self) -> None:
        """Report all detected graph breaks."""
        total_breaks = len(self.graph_break_stmts) + len(self.side_effect_stmts)
        if total_breaks > 0:
            print(f"\n=== SUMMARY: Found {total_breaks} statement(s) with dynamo graph breaks ===")
            if self.graph_break_stmts:
                print(f"\n  Dynamic Control Flow Breaks ({len(self.graph_break_stmts)}):")
                for idx, stmt in enumerate(self.graph_break_stmts, 1):
                    print(f"    {idx}. IfStmt at line {stmt.loc.first_line}: has_break_dyn_cf={getattr(stmt, 'has_break_dyn_cf', False)}")
            if self.side_effect_stmts:
                print(f"\n  Side Effect Breaks ({len(self.side_effect_stmts)}):")
                for idx, stmt in enumerate(self.side_effect_stmts, 1):
                    stmt_type = type(stmt).__name__
                    print(f"    {idx}. {stmt_type} at line {stmt.loc.first_line}: has_break_se={getattr(stmt, 'has_break_se', False)}")
        else:
            print("\n=== SUMMARY: No dynamo graph breaks detected ===")

    def gather_external_symbols(self, node: uni.IfStmt | uni.ElseIf) -> set[uni.Symbol]:
        """Gather external symbols used in this expression statement."""
        name_atoms = node.condition.get_all_sub_nodes(uni.Name)
        symbols = set()
        for name in name_atoms:
            new_symbol = node.sym_tab.lookup(name=name.value, deep=True)
            if new_symbol:
                symbols.add(new_symbol)
        return symbols

    def check_symbol_has_method_call_in_expr(self, expr: uni.Expr, symbol: uni.Symbol, method_name: str) -> bool:
        """Check if a symbol is used with a specific method call in an expression (e.g., b.sum())."""
        # Find all AtomTrailer nodes in the expression
        atom_trailers = expr.get_all_sub_nodes(uni.AtomTrailer)

        for trailer in atom_trailers:
            # Check if this is an attribute access (method call)
            if trailer.is_attr:
                # If target is a Name itself, use it directly
                if isinstance(trailer.target, uni.Name):
                    if trailer.target.value == symbol.sym_name:
                        # Check if the right side (method name) matches
                        if isinstance(trailer.right, uni.Name) and trailer.right.value == method_name:
                            return True
                        # Also check subnodes
                        right_names = trailer.right.get_all_sub_nodes(uni.Name) if hasattr(trailer.right, 'get_all_sub_nodes') else []
                        for method_name_node in right_names:
                            if method_name_node.value == method_name:
                                return True
                else:
                    # Get all names in the target (left side of the dot)
                    target_names = trailer.target.get_all_sub_nodes(uni.Name)
                    # Check if any of the target names match our symbol
                    for target_name in target_names:
                        if target_name.value == symbol.sym_name:
                            # Check if the right side (method name) matches
                            if isinstance(trailer.right, uni.Name) and trailer.right.value == method_name:
                                return True
                            right_names = trailer.right.get_all_sub_nodes(uni.Name) if hasattr(trailer.right, 'get_all_sub_nodes') else []
                            for method_name_node in right_names:
                                if method_name_node.value == method_name:
                                    return True
        return False

    def check_symbol_has_method_call(self, symbol: uni.Symbol, method_name: str) -> bool:
        """Check if symbol's definition expression contains a specific method call (e.g., .sum())."""
        # Get the definition node (NameAtom)
        if not symbol.defn:
            return False
        defn_node = symbol.defn[-1]  # Get the most recent definition

        # Navigate up to find the Assignment statement
        current = defn_node.parent
        while current and not isinstance(current, uni.Assignment):
            current = current.parent

        if not isinstance(current, uni.Assignment):
            return False

        # Check the assignment value for AtomTrailer nodes with the method call
        if current.value:
            atom_trailers = current.value.get_all_sub_nodes(uni.AtomTrailer)
            for trailer in atom_trailers:
                if trailer.is_attr and isinstance(trailer.right, uni.AtomExpr):
                    # Check if right side is a function call with the method name
                    names = trailer.right.get_all_sub_nodes(uni.Name)
                    for name in names:
                        if name.value == method_name:
                            return True

        return False

    def is_symbol_function_parameter(self, symbol: uni.Symbol) -> bool:
        """Check if a symbol is a function parameter (argument)."""
        # Check if the symbol is defined in a function's parameter list
        if not symbol.defn:
            return False

        defn_node = symbol.defn[0]  # Get the original definition

        # Navigate up to find if it's in a ParamVar context
        current = defn_node.parent
        while current:
            # Check if it's a ParamVar (function parameter)
            if isinstance(current, uni.ParamVar):
                return True
            if isinstance(current, uni.Ability):
                # Reached function level without finding ParamVar
                break
            current = current.parent

        return False

    def check_side_effect_call(self, node: uni.UniNode) -> tuple[bool, str]:
        """
        Check if a node contains side-effect causing function calls.
        Returns (has_side_effect, reason).
        """
        # List of built-in functions and modules that cause graph breaks
        side_effect_funcs = {
            "print": "I/O operation",
            "input": "I/O operation",
            "open": "file I/O",
            "write": "I/O operation",
            "read": "I/O operation",
        }
        
        # Logging-related patterns
        logging_patterns = ["log", "logger", "logging", "warn", "error", "info", "debug"]
        
        # Get all function calls in the node
        func_calls = node.get_all_sub_nodes(uni.FuncCall)
        
        for call in func_calls:
            # Check direct function name
            if isinstance(call.target, uni.Name):
                func_name = call.target.value
                
                # Check if it's a built-in function name
                if func_name in side_effect_funcs:
                    # Verify it's actually the built-in, not a redefined variable
                    # Look up the symbol in the symbol table
                    if hasattr(node, 'sym_tab') and node.sym_tab:
                        symbol = node.sym_tab.lookup(name=func_name, deep=True)
                        # If symbol is found and it has a definition, it's user-defined
                        if symbol and symbol.defn:
                            # User has redefined this name, skip it
                            continue
                    # It's the built-in function
                    return True, f"calls {func_name}() ({side_effect_funcs[func_name]})"
                
                # Check for logging patterns
                for pattern in logging_patterns:
                    if pattern in func_name.lower():
                        # For logging, we're less strict since it's less likely to be redefined
                        return True, f"calls {func_name}() (logging operation)"
            
            # Check for method calls like logger.info(), logging.debug(), etc.
            elif isinstance(call.target, uni.AtomTrailer) and call.target.is_attr:
                if isinstance(call.target.right, uni.Name):
                    method_name = call.target.right.value
                    # Check if it's a logging method
                    for pattern in logging_patterns:
                        if pattern in method_name.lower():
                            return True, f"calls .{method_name}() (logging operation)"
                    
                    # Check target for logging modules
                    if isinstance(call.target.target, uni.Name):
                        target_name = call.target.target.value
                        for pattern in logging_patterns:
                            if pattern in target_name.lower():
                                return True, f"calls {target_name}.{method_name}() (logging operation)"
        
        return False, ""

    def trace_symbol_dependencies(self, symbol: uni.Symbol, visited: set[str] | None = None, depth: int = 0) -> tuple[bool, str]:
        """
        Recursively trace a symbol's dependencies to find graph-breaking operations.
        Returns (has_graph_break, reason).
        """
        if visited is None:
            visited = set()

        # Prevent infinite recursion
        if symbol.sym_name in visited or depth > 10:
            return False, ""

        visited.add(symbol.sym_name)
        indent = "  " * depth

        print(f"{indent}Tracing symbol: {symbol.sym_name}")

        # Check if symbol is a function parameter (dynamic value)
        if self.is_symbol_function_parameter(symbol):
            print(f"{indent}  -> Is function parameter (dynamic)")
            return True, f"depends on function parameter '{symbol.sym_name}'"

        # Check if symbol's definition contains graph-breaking operations
        if not symbol.defn:
            return False, ""
        
        defn_node = symbol.defn[-1]
        
        # Navigate up to find the Assignment statement
        current = defn_node.parent
        while current and not isinstance(current, uni.Assignment):
            current = current.parent
        
        if not isinstance(current, uni.Assignment) or not current.value:
            return False, ""
        
        # Check for torch operations that cause graph breaks
        atom_trailers = current.value.get_all_sub_nodes(uni.AtomTrailer)
        for trailer in atom_trailers:
            if trailer.is_attr:
                # Check for torch.max, torch.min, torch.sum, etc.
                target_names = []
                if isinstance(trailer.target, uni.Name):
                    target_names = [trailer.target]
                else:
                    target_names = trailer.target.get_all_sub_nodes(uni.Name)
                
                for target_name in target_names:
                    if target_name.value == "torch":
                        method_name = ""
                        if isinstance(trailer.right, uni.Name):
                            method_name = trailer.right.value
                        
                        # List of torch operations that cause graph breaks
                        graph_breaking_ops = ["max", "min", "sum"]
                        if method_name in graph_breaking_ops:
                            print(f"{indent}  -> Contains torch.{method_name}() (graph-breaking op)")
                            return True, f"uses torch.{method_name}()"
        
        # Recursively check all symbols used in the definition
        name_nodes = current.value.get_all_sub_nodes(uni.Name)
        for name_node in name_nodes:
            # Look up the symbol
            dep_symbol = current.sym_tab.lookup(name=name_node.value, deep=True)
            if dep_symbol and dep_symbol.sym_name not in visited:
                print(f"{indent}  -> Depends on: {dep_symbol.sym_name}")
                has_break, reason = self.trace_symbol_dependencies(dep_symbol, visited, depth + 1)
                if has_break:
                    return True, f"depends on '{dep_symbol.sym_name}' which {reason}"
        
        return False, ""

    def analysis_on_stmt(self, stmt: uni.UniCFGNode) -> None:
        """Perform analysis."""
        # Skip checking entire Ability nodes - we want to check individual statements inside
        if isinstance(stmt, uni.Ability):
            return
        
        # Check for side-effect causing calls in statements
        has_side_effect, se_reason = self.check_side_effect_call(stmt)
        if has_side_effect:
            stmt_id = id(stmt)
            if stmt_id not in self.analyzed_stmts:
                self.analyzed_stmts.add(stmt_id)
                self.side_effect_stmts.append(stmt)
                stmt.has_break_se = True  # type: ignore
                stmt.side_effect_reason = se_reason  # type: ignore
                print(f"\n  *** SIDE EFFECT BREAK DETECTED - Marked {type(stmt).__name__} at line {stmt.loc.first_line} ***")
                print(f"  Reason: {se_reason}")
        
        # Check for dynamic control flow breaks in if statements
        if isinstance(stmt, uni.IfStmt):
            # Check if we've already analyzed this statement (avoid duplicates in CFG traversal)
            stmt_id = id(stmt)
            if stmt_id in self.analyzed_stmts:
                return
            self.analyzed_stmts.add(stmt_id)
            
            symbols = self.gather_external_symbols(stmt)
            print(f"\nExternal symbols in IfStmt at line {stmt.loc.first_line}: {symbols}")
            
            has_graph_break = False
            graph_break_reasons = []
            
            for sym in symbols:
                print(f"\nAnalyzing symbol: {sym.sym_name}, Type: {sym.sym_type}")
                
                # Check if symbol is used with .sum() in the condition
                has_sum_in_condition = self.check_symbol_has_method_call_in_expr(stmt.condition, sym, "sum")
                if has_sum_in_condition:
                    reason = f"Symbol '{sym.sym_name}' is used with .sum() in the condition"
                    print(f"  -> {reason}")
                    has_graph_break = True
                    graph_break_reasons.append(reason)
                    continue  # No need to trace further
                
                # Check if symbol's definition contains .sum()
                has_sum_in_defn = self.check_symbol_has_method_call(sym, "sum")
                if has_sum_in_defn:
                    reason = f"Symbol '{sym.sym_name}' contains .sum() in its definition"
                    print(f"  -> {reason}")
                    has_graph_break = True
                    graph_break_reasons.append(reason)
                    continue
                
                # Trace symbol dependencies recursively
                print(f"  -> Tracing dependencies for '{sym.sym_name}':")
                has_break, trace_reason = self.trace_symbol_dependencies(sym)
                if has_break:
                    reason = f"Symbol '{sym.sym_name}' {trace_reason}"
                    print(f"  -> GRAPH BREAK: {reason}")
                    has_graph_break = True
                    graph_break_reasons.append(reason)
            
            # Mark the statement with a graph break label if detected
            if has_graph_break:
                self.graph_break_stmts.append(stmt)
                # Add custom attributes to mark this statement
                stmt.has_break_dyn_cf = True  # type: ignore
                stmt.dyn_cf_reasons = graph_break_reasons  # type: ignore
                # Keep old name for backward compatibility with FixBreaksPass
                stmt.has_dynamo_graph_break = True  # type: ignore
                stmt.graph_break_reasons = graph_break_reasons  # type: ignore
                print(f"\n  *** DYNAMIC CONTROL FLOW BREAK DETECTED - Marked IfStmt at line {stmt.loc.first_line} ***")
                print(f"  Reasons: {'; '.join(graph_break_reasons)}")
