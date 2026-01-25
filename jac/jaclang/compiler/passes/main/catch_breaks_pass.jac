"""Pytorch Fix Pass."""
import jaclang.pycore.unitree as uni;
import from jaclang.pycore.passes.uni_pass { UniPass }
"""Catch Breaks Pass to handle breaks in loops."""
class CatchBreaksPass(UniPass) {
    """Before pass setup."""
    def before_pass(self: CatchBreaksPass) -> object {
        self._torch_compiled_abilities: list[uni.Ability] = [];
        self._torch_compiled_archetypes: list[uni.Archetype] = [];
        self._current_archetype: (uni.Archetype | None) = None;
        return super.before_pass();
    }
    """Enter archetype (class)."""
    def enter_archetype(self: CatchBreaksPass, <>node: uni.Archetype) -> None {
        self._current_archetype = <>node;
        if <>node.decorators {
            for dec in <>node.decorators {
                dec_txt = dec.unparse().strip() if dec?.unparse else '';
                if (
                    (dec_txt == 'torch.compile')
                    or dec_txt.startswith('torch.compile (')
                ) {
                    self._torch_compiled_archetypes.append(<>node);
                    break;
                }
            }
        }
    }
    """Exit archetype (class)."""
    def exit_archetype(self: CatchBreaksPass, <>node: uni.Archetype) -> None {
        if (self._current_archetype == <>node) {
            self._current_archetype = None;
        }
    }
    """Enter ability (function/method)."""
    def enter_ability(self: CatchBreaksPass, <>node: uni.Ability) -> None {
        should_analyze = False;
        if <>node.decorators {
            for dec in <>node.decorators {
                dec_txt = dec.unparse().strip() if dec?.unparse else '';
                if (
                    (dec_txt == 'torch.compile')
                    or dec_txt.startswith('torch.compile (')
                ) {
                    self._torch_compiled_abilities.append(<>node);
                    should_analyze = True;
                    break;
                }
            }
        }
        if (
            not should_analyze
            and (self._current_archetype in self._torch_compiled_archetypes)
        ) {
            should_analyze = True;
        }
        if should_analyze {
            _ = BreakFinder(<>node);
        }
    }
}
"""CFG Tracer to trace control flow graphs."""
class CFGTracer {
    """Initialize CFGTracer."""
    def init(self: CFGTracer, ability: uni.Ability) -> object {
        self.ability = ability;
        self.run_analysis(self.ability);
    }
    """Analyze symbols."""
    def run_analysis(self: CFGTracer, current_node: uni.UniCFGNode) -> None {
        self.analysis_on_basicblock(current_node);
        if (current_node.get_tail().bb_out == []) {
            return;
        }
        for succ in current_node.get_tail().bb_out {
            self.run_analysis(succ);
        }
    }
    """Perform analysis."""
    def analysis_on_basicblock(self: CFGTracer, current_node: uni.UniCFGNode) -> None {
        basicblock = current_node.get_current_bb();
        for stmt in basicblock {
            self.analysis_on_stmt(stmt);
        }
    }
    """Perform analysis."""
    def analysis_on_stmt(self: CFGTracer, stmt: uni.UniCFGNode) -> None { }
}
"""Break Finder to find break statements."""
class BreakFinder(CFGTracer) {
    with entry {
        _SIDE_EFFECT_CALLS = {'print','input'};
        _LOGGING_PATTERNS = [
            'log',
            'logger',
            'logging',
            'warn',
            'error',
            'info',
            'debug'
        ];
    }

    """Initialize BreakFinder."""
    def init(self: BreakFinder, ability: uni.Ability) -> object {
        self.breaks: list = [];
        self.graph_break_stmts: list[uni.IfStmt] = [];
        self.side_effect_stmts: list[uni.UniNode] = [];
        self.analyzed_stmts: set[int] = <>set();
        super.init(ability);
        self.report_graph_breaks();
    }
    """Report all detected graph breaks."""
    def report_graph_breaks(self: BreakFinder) -> None {
        total_breaks = len(self.graph_break_stmts) + len(self.side_effect_stmts);
        if (total_breaks > 0) {
            ;
        }
    }
    """Gather external symbols used in this expression statement."""
    def gather_external_symbols(
        self: BreakFinder, <>node: (uni.IfStmt | uni.ElseIf)
    ) -> set[uni.Symbol] {
        name_atoms = <>node.condition.get_all_sub_nodes(uni.Name);
        symbols = <>set();
        for name in name_atoms {
            new_symbol = <>node.sym_tab.lookup(name=name.value, deep=True);
            if new_symbol {
                symbols.add(new_symbol);
            }
        }
        return symbols;
    }
    """Check if a symbol is used with a specific method call in an expression (e.g., b.sum())."""
    def check_symbol_has_method_call_in_expr(
        self: BreakFinder, expr: uni.Expr, symbol: uni.Symbol, method_name: str
    ) -> bool {
        atom_trailers = expr.get_all_sub_nodes(uni.AtomTrailer);
        for trailer in atom_trailers {
            if trailer.is_attr {
                if isinstance(trailer.target, uni.Name) {
                    if (trailer.target.value == symbol.sym_name) {
                        if (
                            isinstance(trailer.right, uni.Name)
                            and (trailer.right.value == method_name)
                        ) {
                            return True;
                        }
                        right_names = trailer.right.get_all_sub_nodes(uni.Name)
                        if trailer.right?.get_all_sub_nodes
                        else [];
                        for method_name_node in right_names {
                            if (method_name_node.value == method_name) {
                                return True;
                            }
                        }
                    }
                } else {
                    target_names = trailer.target.get_all_sub_nodes(uni.Name);
                    for target_name in target_names {
                        if (target_name.value == symbol.sym_name) {
                            if (
                                isinstance(trailer.right, uni.Name)
                                and (trailer.right.value == method_name)
                            ) {
                                return True;
                            }
                            right_names = trailer.right.get_all_sub_nodes(uni.Name)
                            if trailer.right?.get_all_sub_nodes
                            else [];
                            for method_name_node in right_names {
                                if (method_name_node.value == method_name) {
                                    return True;
                                }
                            }
                        }
                    }
                }
            }
        }
        return False;
    }
    """Check if symbol's definition expression contains a specific method call (e.g., .sum())."""
    def check_symbol_has_method_call(
        self: BreakFinder, symbol: uni.Symbol, method_name: str
    ) -> bool {
        if not symbol.defn {
            return False;
        }
        defn_node = symbol.defn[-1];
        current = defn_node.parent;
        while (current and not isinstance(current, uni.Assignment)) {
            current = current.parent;
        }
        if not isinstance(current, uni.Assignment) {
            return False;
        }
        if current.value {
            atom_trailers = current.value.get_all_sub_nodes(uni.AtomTrailer);
            for trailer in atom_trailers {
                if (trailer.is_attr and isinstance(trailer.right, uni.AtomExpr)) {
                    names = trailer.right.get_all_sub_nodes(uni.Name);
                    for name in names {
                        if (name.value == method_name) {
                            return True;
                        }
                    }
                }
            }
        }
        return False;
    }
    """Check if a symbol is a function parameter (argument)."""
    def is_symbol_function_parameter(self: BreakFinder, symbol: uni.Symbol) -> bool {
        if not symbol.defn {
            return False;
        }
        defn_node = symbol.defn[0];
        current = defn_node.parent;
        while current {
            if isinstance(current, uni.ParamVar) {
                return True;
            }
            if isinstance(current, uni.Ability) {
                break;
            }
            current = current.parent;
        }
        return False;
    }
    """
        Check if a node contains side-effect causing function calls.
        Returns (has_side_effect, reason).
        """
    def check_side_effect_call(
        self: BreakFinder, <>node: uni.UniNode
    ) -> tuple[bool, str] {
        func_calls = <>node.get_all_sub_nodes(uni.FuncCall);
        for call in func_calls {
            (is_se, reason) = self._is_side_effect_call(call);
            if is_se {
                return (True, reason);
            }
        }
        return (False, '');
    }

    """Check if a function call has side effects and should be buffered."""
    def _is_side_effect_call(
        self: BreakFinder, <>node: uni.FuncCall
    ) -> tuple[bool, str] {
        if isinstance(<>node.target, uni.Name) {
            func_name = <>node.target.value;
            if (func_name in self._SIDE_EFFECT_CALLS) {
                if (<>node?.sym_tab and <>node.sym_tab) {
                    symbol = <>node.sym_tab.lookup(name=func_name, deep=True);
                    if (symbol and symbol.defn) {
                        return (False, '');
                    }
                }
                return (True, f"calls {func_name}() (I/O operation)");
            }
            for pattern in self._LOGGING_PATTERNS {
                if (pattern in func_name.lower()) {
                    return (True, f"calls {func_name}() (logging operation)");
                }
            }
        } elif isinstance(<>node.target, uni.AtomTrailer) {
            parts = [];
            current = <>node.target;
            while isinstance(current, uni.AtomTrailer) {
                if (current?.right and isinstance(current.right, uni.Name)) {
                    parts.append(current.right.value);
                }
                current = current.target;
                if isinstance(current, uni.Name) {
                    parts.append(current.value);
                    break;
                }
            }
            for part in parts {
                for pattern in self._LOGGING_PATTERNS {
                    if (pattern in part.lower()) {
                        return (
                            True,
                            f"calls {'.'.join(reversed(parts))}() (logging operation)"
                        );
                    }
                }
            }
        }
        return (False, '');
    }
    """
        Recursively trace a symbol's dependencies to find graph-breaking operations.
        Returns (has_graph_break, reason).
        """
    def trace_symbol_dependencies(
        self: BreakFinder,
        symbol: uni.Symbol,
        visited: (set[str] | None) = None,
        depth: int = 0
    ) -> tuple[bool, str] {
        if (visited is None) {
            visited = <>set();
        }
        if ((symbol.sym_name in visited) or (depth > 10)) {
            return (False, '');
        }
        visited.add(symbol.sym_name);
        if self.is_symbol_function_parameter(symbol) {
            return (True, f"depends on function parameter '{symbol.sym_name}'");
        }
        if not symbol.defn {
            return (False, '');
        }
        defn_node = symbol.defn[-1];
        current = defn_node.parent;
        while (current and not isinstance(current, uni.Assignment)) {
            current = current.parent;
        }
        if (not isinstance(current, uni.Assignment) or not current.value) {
            return (False, '');
        }
        atom_trailers = current.value.get_all_sub_nodes(uni.AtomTrailer);
        for trailer in atom_trailers {
            if trailer.is_attr {
                target_names = [];
                if isinstance(trailer.target, uni.Name) {
                    target_names = [trailer.target];
                } else {
                    target_names = trailer.target.get_all_sub_nodes(uni.Name);
                }
                for target_name in target_names {
                    if (target_name.value == 'torch') {
                        method_name = '';
                        if isinstance(trailer.right, uni.Name) {
                            method_name = trailer.right.value;
                        }
                        graph_breaking_ops = ['max', 'min', 'sum'];
                        if (method_name in graph_breaking_ops) {
                            return (True, f"uses torch.{method_name}()");
                        }
                    }
                }
            }
        }
        name_nodes = current.value.get_all_sub_nodes(uni.Name);
        for name_node in name_nodes {
            dep_symbol = current.sym_tab.lookup(name=name_node.value, deep=True);
            if (dep_symbol and (dep_symbol.sym_name not in visited)) {
                (has_break, reason) = self.trace_symbol_dependencies(
                    dep_symbol, visited, (depth + 1)
                );
                if has_break {
                    return (True, f"depends on '{dep_symbol.sym_name}' which {reason}");
                }
            }
        }
        return (False, '');
    }

    """Perform analysis."""
    def analysis_on_stmt(self: BreakFinder, stmt: uni.UniCFGNode) -> None {
        if isinstance(stmt, uni.Ability) {
            return;
        }
        (has_side_effect, se_reason) = self.check_side_effect_call(stmt);
        if has_side_effect {
            stmt_id = id(stmt);
            if (stmt_id not in self.analyzed_stmts) {
                self.analyzed_stmts.add(stmt_id);
                self.side_effect_stmts.append(stmt);
                func_calls = stmt.get_all_sub_nodes(uni.FuncCall);
                for call in func_calls {
                    (is_se, _) = self._is_side_effect_call(call);
                    if is_se {
                        call.has_break_se = True;
                        call.side_effect_reason = se_reason;
                    }
                }
                stmt.has_break_se = True;
                stmt.side_effect_reason = se_reason;
            }
        }
        if isinstance(stmt, uni.IfStmt) {
            stmt_id = id(stmt);
            if (stmt_id in self.analyzed_stmts) {
                return;
            }
            self.analyzed_stmts.add(stmt_id);
            symbols = self.gather_external_symbols(stmt);
            has_graph_break = False;
            graph_break_reasons = [];
            for sym in symbols {
                has_sum_in_condition = self.check_symbol_has_method_call_in_expr(
                    stmt.condition, sym, 'sum'
                );
                if has_sum_in_condition {
                    reason = f"Symbol '{sym.sym_name}' is used with .sum() in the condition";
                    has_graph_break = True;
                    graph_break_reasons.append(reason);
                    continue;
                }
                has_sum_in_defn = self.check_symbol_has_method_call(sym, 'sum');
                if has_sum_in_defn {
                    reason = f"Symbol '{sym.sym_name}' contains .sum() in its definition";
                    has_graph_break = True;
                    graph_break_reasons.append(reason);
                    continue;
                }
                (has_break, trace_reason) = self.trace_symbol_dependencies(sym);
                if has_break {
                    reason = f"Symbol '{sym.sym_name}' {trace_reason}";
                    has_graph_break = True;
                    graph_break_reasons.append(reason);
                }
            }
            if has_graph_break {
                self.graph_break_stmts.append(stmt);
                stmt.has_break_dyn_cf = True;
                stmt.dyn_cf_reasons = graph_break_reasons;
                stmt.has_dynamo_graph_break = True;
                stmt.graph_break_reasons = graph_break_reasons;
            }
        }
    }
}

