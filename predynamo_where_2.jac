import torch;
class Cfg {
    def init(self: Cfg, max_position_embeddings: int = 8, has_original: bool = False) {
        self.max_position_embeddings = max_position_embeddings;
        if has_original {
            self.original_max_position_embeddings = max_position_embeddings // 2;
        }
    }
}

@torch.compile()
class ToyModel(torch.nn.Module) {
    def init(self: ToyModel, cfg: Cfg) {
        super.init();
        self.config = cfg;
        self.long_inv_freq = torch.tensor([10.0, 20.0, 30.0]);
        self.original_inv_freq = torch.tensor([1.0, 2.0, 3.0]);
    }

    def rope_init_fn(self: ToyModel, cfg: Any, device: Any, seq_len: int) {
        return (torch.arange(1, 4, dtype=torch.float32, device=device), None);
    }

    def _longrope_frequency_update(
        self: ToyModel, position_ids: Any, device: Any = 'cpu'
    ) {
        seq_len = torch.max(position_ids) + 1;
        if self.config?.original_max_position_embeddings {
            original_max_position_embeddings = self.config.original_max_position_embeddings;
        } else {
            original_max_position_embeddings = self.config.max_position_embeddings;
        }
        if (seq_len > original_max_position_embeddings) {
            self.register_buffer('inv_freq', self.long_inv_freq, persistent=False);
        } else {
            self.register_buffer(
                'inv_freq', self.original_inv_freq.to(device), persistent=False
            );
        }
        return self.inv_freq;
    }
}

with entry {
    # Create a config and model instance
    cfg = Cfg(8, True);
    model = ToyModel(cfg);

    # Create test inputs
    position_ids = torch.tensor([0, 1, 2, 3]);
    device = 'cpu';

    # Use torch._dynamo.explain to check for graph breaks
    print("=" * 80);
    print("Testing _longrope_frequency_update with torch._dynamo.explain");
    print("=" * 80);
    explain_output = torch._dynamo.explain(model._longrope_frequency_update)(position_ids, device);

    print("\nGraph Break Analysis:");
    print(explain_output);

    # Check for graph breaks
    if hasattr(explain_output, 'graph_break_count') and explain_output.graph_break_count > 0 {
        print("\n" + "=" * 80);
        print(f"⚠ GRAPH BREAKS DETECTED: {explain_output.graph_break_count}");
        print("=" * 80);
    } else {
        print("\n" + "=" * 80);
        print("✓ No graph breaks detected!");
        print("=" * 80);
    }
}
