"""Literal paper Fig. 4 (Phi-4 longrope): multi-statement data-dependent
branches that share a trailing register_buffer call, with a nested *static*
hasattr guard that must be preserved and hoisted, not rewritten."""
import torch;

@torch.compile()
def update(self_obj: any, position_ids: any, device: any) -> any {
    seq_len = torch.max(position_ids) + 1;
    if hasattr(self_obj.config, "original_max_position_embeddings") {
        omp = self_obj.config.original_max_position_embeddings;
    } else {
        omp = self_obj.config.max_position_embeddings;
    }
    if (seq_len > omp) {
        if not hasattr(self_obj, "long_inv_freq") {
            self_obj.long_inv_freq = self_obj.rope_init_fn(self_obj.config, device);
        }
        self_obj.register_buffer("inv_freq", self_obj.long_inv_freq, persistent=False);
    } else {
        self_obj.original_inv_freq = self_obj.original_inv_freq.to(device);
        self_obj.register_buffer("inv_freq", self_obj.original_inv_freq, persistent=False);
    }
    return self_obj.inv_freq;
}
