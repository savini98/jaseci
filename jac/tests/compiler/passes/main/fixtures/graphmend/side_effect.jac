"""Python side effect inside a compiled region: print -> deferred buffer."""
import torch;

@torch.compile()
def f(a: int) -> int {
    print(a);
    a = a + 1;
    return a;
}
