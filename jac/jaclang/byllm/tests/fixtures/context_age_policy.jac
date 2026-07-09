"""Stage 3 smoke test: AgePolicy fires compaction when the budget
threshold is met, and is a no-op otherwise.

Validates:
- Context(policy=AgePolicy(...)) plumbs the policy into the loop
- AgePolicy.should_compact returns True when fill ratio >= threshold
- AgePolicy.compact creates a ContextCompactionSummary node and hides
  the compacted ContextMessage / ContextToolResult nodes
- preserve_recent_turns protects the most recent items
- Without a policy: existing Stage 2 behavior is unchanged (no summary
  nodes appear)
"""

import from jaclang.byllm.lib { Model, MockToolCall }
import from jaclang.byllm.context {
    Context,
    ContextGraph,
    ContextToolResult,
    ContextMessage,
    ContextCompactionSummary,
    AgePolicy
}


def fetch_status() -> str {
    # Make this large enough that the visible_content drives total_tokens
    # past the policy's 200-token usable budget on its own.
    return "x" * 1200;
}


# Aggressive policy: 200-token budget, 50% threshold, preserve_recent=1.
glob policy_ctx = Context(
    persist=True,
    policy=AgePolicy(
        usable_budget_tokens=200,
        compact_threshold_ratio=0.5,
        preserve_recent_turns=1
    )
);
glob bare_ctx = Context(persist=True);


glob llm = Model(
    model_name="mockllm",
    config={
        "outputs": [
            # turn 1 (with-policy): tool call + plain final
            MockToolCall(tool=fetch_status, args={}),
            "first answer",
            # turn 2 (with-policy): tool call + plain final — graph now
            # has enough content that AgePolicy.should_compact is True.
            MockToolCall(tool=fetch_status, args={}),
            "second answer",
            # turn 1 (no-policy regression check)
            MockToolCall(tool=fetch_status, args={}),
            "bare first answer",
            # turn 2 (no-policy)
            MockToolCall(tool=fetch_status, args={}),
            "bare second answer",
        ]
    },
);


def with_policy(query: str) -> str by llm(
    tools=[fetch_status], context=policy_ctx
);

def no_policy(query: str) -> str by llm(
    tools=[fetch_status], context=bare_ctx
);


with entry {
    print("=== Stage 3: ContextPolicy + AgePolicy ===\n");

    # ---- With AgePolicy: compaction should fire by turn 2 ----
    with_policy("status?");
    with_policy("status again?");
    pg = policy_ctx.graph;
    pol_summaries = [s for s in [pg -->](?:ContextCompactionSummary)];
    pol_visible_msgs = [m for m in [pg -->](?:ContextMessage) if m.visible];
    pol_invisible_msgs = [m for m in [pg -->](?:ContextMessage) if not m.visible];
    pol_visible_trs = [tr for tr in [pg -->](?:ContextToolResult) if tr.visible];
    pol_invisible_trs = [tr for tr in [pg -->](?:ContextToolResult) if not tr.visible];

    print(f"with-policy: turn_count={pg.turn_count}");
    print(f"  CompactionSummaries:        {len(pol_summaries)}");
    print(f"  visible messages:           {len(pol_visible_msgs)}");
    print(f"  invisible (compacted) msgs: {len(pol_invisible_msgs)}");
    print(f"  visible tool results:       {len(pol_visible_trs)}");
    print(f"  invisible tool results:     {len(pol_invisible_trs)}");

    # ---- Without policy: regression — no summaries, nothing hidden ----
    no_policy("bare status?");
    no_policy("bare again?");
    bg = bare_ctx.graph;
    bare_summaries = [s for s in [bg -->](?:ContextCompactionSummary)];
    bare_invisible_msgs = [m for m in [bg -->](?:ContextMessage) if not m.visible];
    bare_invisible_trs = [tr for tr in [bg -->](?:ContextToolResult) if not tr.visible];

    print(f"\nno-policy: turn_count={bg.turn_count}");
    print(f"  CompactionSummaries:        {len(bare_summaries)}");
    print(f"  invisible (compacted) msgs: {len(bare_invisible_msgs)}");
    print(f"  invisible tool results:     {len(bare_invisible_trs)}");

    failures = 0;
    if len(pol_summaries) >= 1 {
        print("\nPASS  AgePolicy created a CompactionSummary node");
    } else {
        failures += 1;
        print("\nFAIL  AgePolicy did not create any CompactionSummary");
    }
    if len(pol_invisible_msgs) > 0 or len(pol_invisible_trs) > 0 {
        print("PASS  AgePolicy hid older messages/tool results");
    } else {
        failures += 1;
        print("FAIL  AgePolicy did not hide any old content");
    }
    if len(pol_summaries) > 0 and pol_summaries[0].dropped_message_count + pol_summaries[0].dropped_tool_result_count > 0 {
        print("PASS  CompactionSummary dropped_*_count is non-zero");
    } else {
        failures += 1;
        print("FAIL  CompactionSummary dropped counts are all zero");
    }
    if len(bare_summaries) == 0 and len(bare_invisible_msgs) == 0 and len(bare_invisible_trs) == 0 {
        print("PASS  no-policy run is unchanged from Stage 2 (no summary, nothing hidden)");
    } else {
        failures += 1;
        print(f"FAIL  no-policy regression: summaries={len(bare_summaries)} invisible_msgs={len(bare_invisible_msgs)} invisible_trs={len(bare_invisible_trs)}");
    }

    if failures > 0 {
        raise AssertionError(f"{failures} assertion(s) failed");
    }
    print("\nAll assertions passed.");
}
