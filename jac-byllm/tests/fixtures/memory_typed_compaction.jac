"""Typed-slot compaction smoke test.

Validates that `TypedSlotCompactionPolicy.compact` produces a structured
summary that preserves verbatim numeric values — the failure mode that
direction-1-results-v2.md Phase C.3 measured against the free-form
AgePolicy.

The fixture builds a MemoryGraph by hand (skipping the ReAct loop) so
the call to `policy.compact(graph, None)` is deterministic and the
single LLM call we make is `extract_compaction_slots`. MockLLM is
configured to return a pre-defined CompactionSlots instance that
contains the planted numeric value verbatim — what we verify is that:

- compact() returns a non-None MemoryCompactionSummary.
- The summary_text contains the planted value as a substring (so a
  later turn searching for the identifier would still find the
  number).
- The originals are hidden (visible == False).
"""

import time;
import from byllm.lib { Model }
import from byllm.memory {
    Memory,
    MemoryGraph,
    MemoryToolResult,
    MemoryMessage,
    MemoryCompactionSummary
}
import from byllm.typed_compaction {
    CompactionSlots,
    TypedSlotCompactionPolicy
}


# MockLLM returns a CompactionSlots when extract_compaction_slots fires.
# The slots contain the planted value 'keepalive_seconds = 47
# (config/prod.yaml)' verbatim — the byLLM Phase-C.3 free-form summary
# dropped exactly this kind of detail.
glob llm = Model(
    model_name="mockllm",
    config={
        "outputs": [
            CompactionSlots(
                files_read=["config/prod.yaml", "services/api.py"],
                key_observations=[
                    "keepalive_seconds = 47 (config/prod.yaml)",
                    "RATE_LIMIT_BURST_CAPACITY = 83 (services/api.py)"
                ],
                commands_run=["ls -la fixtures/project/"],
                decisions=["proceed with rate-limit defaults"],
                unresolved_questions=[]
            )
        ]
    },
);


with entry {
    print("=== Typed-slot compaction smoke test ===\n");

    # Stand up a MemoryGraph manually with enough older content to
    # compact. preserve_recent_turns=1 means anything at turn <=
    # turn_count - 1 is eligible.
    graph = MemoryGraph(session_id="t1", created_at=time.time());
    root() ++> graph;
    graph.turn_count = 3;   # so turns 1 and 2 are eligible

    # Older message + tool result the extractor will summarise.
    graph.record_message(role="user", turn=1, content="check the config");
    graph.record_message(role="assistant", turn=1, content="opening prod.yaml");
    graph.record_tool_result(
        tool_name="read_file",
        args={"path": "config/prod.yaml"},
        full_content="keepalive_seconds: 47\nbind_port: 8080\n",
        visible_content="",
        turn=1
    );
    graph.record_message(role="user", turn=2, content="and the API");
    graph.record_message(role="assistant", turn=2, content="opening api.py");
    graph.record_tool_result(
        tool_name="read_file",
        args={"path": "services/api.py"},
        full_content="RATE_LIMIT_BURST_CAPACITY = 83\n",
        visible_content="",
        turn=2
    );

    # Sanity: before compaction, originals are visible.
    pre_visible_msgs = [
        m for m in [graph -->][?:MemoryMessage] if m.visible
    ];
    pre_visible_trs = [
        tr for tr in [graph -->][?:MemoryToolResult] if tr.visible
    ];
    print(
        f"pre: visible msgs={len(pre_visible_msgs)} "
        f"trs={len(pre_visible_trs)}"
    );

    # Run typed-slot compaction. mt_run is unused inside compact for
    # this policy, so passing None is fine.
    policy = TypedSlotCompactionPolicy(
        usable_budget_tokens=200,
        compact_threshold_ratio=0.0,
        preserve_recent_turns=1
    );
    summary = policy.compact(graph, None);
    print(f"compact returned: {summary is not None}");
    if summary is not None {
        print("--- summary_text ---");
        print(summary.summary_text);
        print("--- end summary ---");
    }

    # Re-check graph state.
    post_visible_msgs = [
        m for m in [graph -->][?:MemoryMessage] if m.visible
    ];
    post_visible_trs = [
        tr for tr in [graph -->][?:MemoryToolResult] if tr.visible
    ];
    summary_nodes = [
        s for s in [graph -->][?:MemoryCompactionSummary]
    ];
    print(
        f"post: visible msgs={len(post_visible_msgs)} "
        f"trs={len(post_visible_trs)} summaries={len(summary_nodes)}"
    );

    failures = 0;

    if summary is not None {
        print("PASS  compact returned a MemoryCompactionSummary");
    } else {
        failures += 1;
        print("FAIL  compact returned None");
    }

    if (
        summary is not None
        and "keepalive_seconds = 47" in summary.summary_text
    ) {
        print("PASS  summary_text contains 'keepalive_seconds = 47' verbatim");
    } else {
        failures += 1;
        print(
            "FAIL  summary_text missing the keepalive_seconds=47 marker"
        );
    }

    if (
        summary is not None
        and "RATE_LIMIT_BURST_CAPACITY = 83" in summary.summary_text
    ) {
        print(
            "PASS  summary_text contains "
            "'RATE_LIMIT_BURST_CAPACITY = 83' verbatim"
        );
    } else {
        failures += 1;
        print(
            "FAIL  summary_text missing the burst_capacity=83 marker"
        );
    }

    if len(post_visible_msgs) < len(pre_visible_msgs) {
        print("PASS  some MemoryMessages were hidden by compaction");
    } else {
        failures += 1;
        print(
            f"FAIL  visible message count did not drop "
            f"({len(pre_visible_msgs)} -> {len(post_visible_msgs)})"
        );
    }

    if len(post_visible_trs) < len(pre_visible_trs) {
        print("PASS  some MemoryToolResults were hidden by compaction");
    } else {
        failures += 1;
        print(
            f"FAIL  visible tool result count did not drop "
            f"({len(pre_visible_trs)} -> {len(post_visible_trs)})"
        );
    }

    if len(summary_nodes) == 1 {
        print("PASS  exactly one MemoryCompactionSummary node attached");
    } else {
        failures += 1;
        print(
            f"FAIL  expected 1 summary node, got {len(summary_nodes)}"
        );
    }

    if failures > 0 {
        raise AssertionError(f"{failures} assertion(s) failed");
    }
    print("\nAll assertions passed.");
}
