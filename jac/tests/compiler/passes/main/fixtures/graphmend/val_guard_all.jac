"""Validation guard on a tensor-bool op (.all()): asserted directly via
torch._assert_async (no Python-bool conversion needed)."""
import torch;

@torch.compile()
def f(mask: any) -> any {
    if not mask.all() {
        raise ValueError("mask must be all true");
    }
    return mask;
}
