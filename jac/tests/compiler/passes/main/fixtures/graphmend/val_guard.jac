"""Validation guard inside a compiled region: raise -> torch._assert_async."""
import torch;

@torch.compile()
def f(attention_mask: any, expected: any) -> any {
    if not torch.equal(attention_mask, expected) {
        raise ValueError("unsupported attention mask");
    }
    return process(attention_mask);
}
