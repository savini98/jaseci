"""
Fix fx-graph breaks
"""


import jaclang.compiler.unitree as uni
from jaclang.compiler.passes import UniPass

class PreDynamoPass(UniPass):
    """Pre-Dynamo pass to prepare the Jac AST for Dynamo processing."""

    def before_pass(self) -> None:
        """Prepare the AST before the pass."""
        # This is where we can set up any necessary state or checks before processing
        self.breaks:list[uni.UniNode] = []

    def enter_if_stmt(self, node: uni.IfStmt) -> None:
        """Handle entering an IfStmt node."""
        # Check if the IfStmt has a break condition
        # print("Entering IfStmt with condition:", node.condition)
        parent = node.find_parent_of_type(uni.Ability)
        if parent:
            print(">>>>> Found parent Ability:", parent)
            print(self.gather_external_symbols(node))
            # If the parent is an Ability, we can add a break to the list
            func_calls = self.check_for_function_calls(node)
            for func_call in func_calls:
                print("Target:", func_call.target.right)
    
    def gather_external_symbols(self, node:uni.IfStmt | uni.ElseIf) -> set[uni.Symbol]:
        """Gather external symbols used in this expression statement."""
        name_atoms = node.condition.get_all_sub_nodes(uni.Name)
        print(f"names: {name_atoms}")
        symbols = set()
        for name in name_atoms:
            new_symbol = node.sym_tab.lookup(name=name.value, deep=True)
            if new_symbol:
                symbols.add(new_symbol)
        return symbols
    
    def check_for_function_calls(self, node: uni.IfStmt | uni.ElseIf) -> list:
        """Check if the node contains any function calls."""
        all_nodes = node.flatten()
        func_calls = []
        for ast_node in all_nodes:
            if isinstance(ast_node, uni.FuncCall):
                print(f"Found function call: {ast_node}")
                # Here we can add logic to handle the function call
                # For example, we could check if it matches a specific pattern
                func_calls.append(ast_node)
        print(f"FunctionCall nodes: {func_calls}")
        return func_calls
        
            