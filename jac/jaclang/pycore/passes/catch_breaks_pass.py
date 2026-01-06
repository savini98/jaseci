"""Pytorch Fix Pass."""

import jaclang.pycore.unitree as uni
from jaclang.pycore.passes.uni_pass import UniPass


class CatchBreaksPass(UniPass):
    """Catch Breaks Pass to handle breaks in loops."""

    def before_pass(self):
        """Before pass setup."""
        # print("Running CatchBreaksPass...")
        self._torch_compiled_abilities: list[uni.Ability] = []
        return super().before_pass()

    def enter_ability(self, node: uni.Ability) -> None:
        """Enter ability."""
        # print(f"Visiting ability")
        if node.decorators:
            for dec in node.decorators:
                # print(dec.unparse())
                dec_txt = dec.unparse().strip() if hasattr(dec, "unparse") else ""
                if dec_txt == "torch.compile" or dec_txt.startswith(
                    "torch.compile ("
                ):
                    self._torch_compiled_abilities.append(node)
                    tracer = BreakFinder(node)


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
        pass


class BreakFinder(CFGTracer):
    """Break Finder to find break statements."""

    def __init__(self, ability: uni.Ability):
        """Initialize BreakFinder."""
        super().__init__(ability)
        self.breaks: list[uni.BreakStmt] = []

    def gather_external_symbols(self, node: uni.IfStmt | uni.ElseIf) -> set[uni.Symbol]:
        """Gather external symbols used in this expression statement."""
        name_atoms = node.condition.get_all_sub_nodes(uni.Name)
        symbols = set()
        for name in name_atoms:
            new_symbol = node.sym_tab.lookup(name=name.value, deep=True)
            if new_symbol:
                symbols.add(new_symbol)
        return symbols

    def analysis_on_stmt(self, stmt: uni.UniCFGNode) -> None:
        """Perform analysis."""
        if isinstance(stmt, uni.IfStmt):
            symbols = self.gather_external_symbols(stmt)
            print(f"External symbols in IfStmt: {symbols}")
            for sym in symbols:
                print(f"Symbol: {sym.sym_name}, Type: {sym.sym_type}")
