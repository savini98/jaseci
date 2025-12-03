"""Pytorch Fix Pass."""

import jaclang.compiler.unitree as uni
from jaclang.compiler.constant import Tokens as Tok
from jaclang.compiler.passes import UniPass


class CatchBreaksPass(UniPass):
    """Catch Breaks Pass to handle breaks in loops."""

    def before_pass(self):
        """Before pass setup."""
        # print("Running CatchBreaksPass...")
        self._torch_compiled_abilities: list[uni.Ability] = []
        return super().before_pass()

    def enter_ability(self, node: uni.Ability) -> None:
        """Enter ability."""
        if node.decorators:
            for dec in node.decorators:
                dec_txt = dec.unparse().strip() if hasattr(dec, "unparse") else ""
                # Accept forms: torch.compile, torch.compile(...)
                if dec_txt == "torch.compile" or dec_txt.startswith("torch.compile("):
                    self._torch_compiled_abilities.append(node)
                    # Mark node for downstream passes.
                    print(f"Marked ability '{node.name}' for torch compilation.")


