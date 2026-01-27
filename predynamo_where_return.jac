import torch;

@torch.compile()
def with_breaks(a: Any, b: Any) {
    x = a / (torch.abs(a) + 1);
    b = x * 1.414;
    if (b.sum() < 0) {
        return x * (b * -1);
    } else {
        return x * (b * -2);
    } 
}

with entry {
    # Create test inputs
    a = torch.tensor([1.0, 2.0, 3.0]);
    b = torch.tensor([4.0, 5.0, 6.0]);

    # Use torch._dynamo.explain to check for graph breaks
    print("=" * 80);
    print("Testing with_breaks with torch._dynamo.explain");
    print("=" * 80);
    explain_output = torch._dynamo.explain(with_breaks)(a, b);

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
