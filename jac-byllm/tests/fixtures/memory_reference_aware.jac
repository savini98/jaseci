"""Stage 4 smoke test: ReferenceAwarePolicy preserves high-ref content
while evicting low-ref content of the same age and size.

Two MemoryToolResult nodes are created with identical age and size.
One gets a ref_count bump (simulating that the assistant cited its
content). With ReferenceAwarePolicy, the unreferenced one should be
evicted first; the referenced one should survive.

Tests the policy logic directly (no LLM call needed) by constructing a
small MemoryGraph manually and asking the policy what to evict.
"""

import from byllm.memory {
    Memory,
    MemoryGraph,
    MemoryToolResult,
    ReferenceAwarePolicy,
    EvictionDecision,
    apply_eviction_decision
}


with entry {
    print("=== Stage 4: ReferenceAwarePolicy preserves cited content ===\n");

    # Build a graph with two tool results, same age + size.
    g = MemoryGraph(session_id="s4", created_at=0.0, turn_count=5);

    # Roughly 1000 chars each ⇒ ~250 tokens visible. Two of these put
    # tool-result tokens at ~500, comfortably above policy.target.
    big_content = "alpha\nbeta\ngamma\n" + ("x" * 950);

    high_ref = MemoryToolResult(
        tr_id="tr_1_high",
        tool_name="fetch_high",
        full_content=big_content,
        visible_content=big_content,
        produced_at_turn=1,
        reference_count=8,
        last_referenced_at=1.0
    );
    low_ref = MemoryToolResult(
        tr_id="tr_1_low",
        tool_name="fetch_low",
        full_content=big_content,
        visible_content=big_content,
        produced_at_turn=1,
        reference_count=0,
        last_referenced_at=0.0
    );
    g ++> high_ref;
    g ++> low_ref;

    # Budget calibrated so evicting one (low-ref) drops fill below the
    # 0.5 target; that's the boundary the early-stop semantics protect
    # the high-ref candidate at.
    policy = ReferenceAwarePolicy(
        usable_budget_tokens=800,
        soft_threshold_ratio=0.3,
        hard_threshold_ratio=0.5,
        target_ratio=0.5,
        keep_last_turns=1
    );

    should_prune = policy.should_prune(g, None);
    decisions = policy.select_eviction(g, None);
    evicted_ids = [d.tool_result.tr_id for d in decisions];

    print(f"should_prune: {should_prune}");
    print(f"eviction order (lowest score first):");
    for d in decisions {
        print(f"  {d.tool_result.tr_id} ({d.tool_result.tool_name}) kind={d.kind}");

    }
    # Apply the decisions and check what survives.
    for d in decisions {
        apply_eviction_decision(d);
    }
    survivors = [tr.tr_id for tr in [g -->](?:MemoryToolResult) if tr.truncation_state != "hard_cleared" and len(tr.visible_content) > 100];
    print(f"\nsurvivors (still visible with content): {survivors}");

    failures = 0;
    if should_prune {
        print("\nPASS  should_prune True at fill_ratio above soft threshold");
    } else {
        failures += 1;
        print("FAIL  should_prune returned False when budget was exceeded");
    }
    if len(decisions) >= 1 {
        print("PASS  policy returned at least one eviction decision");
    } else {
        failures += 1;
        print("FAIL  policy returned no decisions");
    }
    # The low-ref one must be evicted FIRST (lowest score). The high-ref
    # one might or might not be evicted depending on whether one is enough.
    if len(decisions) >= 1 and decisions[0].tool_result.tr_id == "tr_1_low" {
        print("PASS  unreferenced (low-ref) candidate was evicted first");
    } else {
        failures += 1;
        first = decisions[0].tool_result.tr_id if decisions else "<none>";
        print(f"FAIL  expected first eviction = 'tr_1_low', got {first!r}");
    }
    if "tr_1_high" in survivors {
        print("PASS  high-ref content preserved through pressure");
    } else {
        failures += 1;
        print(f"FAIL  high-ref content not preserved. survivors={survivors}");
    }

    if failures > 0 {
        raise AssertionError(f"{failures} assertion(s) failed");
    }
    print("\nAll assertions passed.");
}
