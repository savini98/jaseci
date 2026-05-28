"""Integration test for the materialize hook.

Confirms that when a Context(policy=...) is wired through `by llm()`,
the framework actually calls `policy.materialize()` to build the
message list that goes to the LLM — i.e. policy decisions reach LLM
context, not just graph state.

Strategy: subclass AgePolicy with a counter that increments inside
`materialize`. Run a small MockLLM-backed sequence, then assert the
counter advanced. Without the wiring in get_msg_list, the counter
stays at zero (the R2 finding, made executable).

A separate scenario then hard-clears a tool-result's visible_content on
the graph and re-runs materialize directly; we verify that the
original value does NOT appear in the captured material output.
(MockLLM doesn't reconstruct LiteLLM-shaped assistant messages with
tool_calls, so the paired ToolCallResultMsg shape isn't exercised
here — see memory_materialize_unit.jac for that path.)
"""

import time;
import from byllm.lib { Model, MockToolCall }
import from byllm.types { Message, MessageRole, ToolCallResultMsg }
import from byllm.context {
    Context,
    ContextGraph,
    ContextToolResult,
    ContextMessage,
    ContextCompactionSummary,
    AgePolicy,
    EvictionDecision,
    apply_eviction_decision
}


# A policy that counts materialize() invocations and also runs
# aggressive pruning so we can verify pruning decisions reach the LLM
# context via materialize.
obj CountingPolicy(AgePolicy) {
    has materialize_count: int = 0,
        last_materialized: list = [];

    override def materialize(
        graph: ContextGraph, mt_run: object
    ) -> list | None;
}


impl CountingPolicy.materialize(
    graph: ContextGraph, mt_run: object
) -> list | None {
    self.materialize_count += 1;
    result = super.materialize(graph, mt_run);
    if result is not None {
        self.last_materialized = result;
    }
    return result;
}


def fetch_config() -> str {
    return (
        "LIGHTHOUSE_PROD_CONFIG\n"
        "MAX_TOKEN_REFRESH_ATTEMPTS = 73\n"
    );
}


glob policy = CountingPolicy(
    usable_budget_tokens=10000,
    compact_threshold_ratio=10.0,  # effectively disable AgePolicy compaction
    preserve_recent_turns=1
);
glob ctx = Context(persist=True, policy=policy);


glob llm = Model(
    model_name="mockllm",
    config={
        "outputs": [
            # Turn 1: tool call returning the planted value, then ack.
            MockToolCall(tool=fetch_config, args={}),
            "ok",
            # Turn 2: ack (no tool call).
            "all set",
        ]
    },
);


def investigate(query: str) -> str by llm(
    tools=[fetch_config], context=ctx
);


def follow_up(query: str) -> str by llm(context=ctx);


with entry {
    print("=== materialize integration test ===\n");

    # Turn 1: tool call + final
    investigate("what's the config?");
    count_after_t1 = policy.materialize_count;
    print(f"materialize calls after turn 1: {count_after_t1}");

    # Locate the recorded tool result and hard-clear its visible_content
    # by hand. This simulates what ReferenceAwarePolicy.select_eviction
    # would have decided on a turn-2 prune cycle. We then run another
    # by-llm call and verify the materialised messages reflect the
    # cleared state — not the original "MAX_TOKEN_REFRESH_ATTEMPTS = 73".
    g = ctx.graph;
    if g is None {
        raise AssertionError("graph not initialised");
    }
    target_tr: ContextToolResult | None = None;
    for tr in [g -->][?:ContextToolResult] {
        if tr.tool_name == "fetch_config" {
            target_tr = tr;
        }
    }
    if target_tr is None {
        raise AssertionError("expected a fetch_config ContextToolResult");
    }
    print(
        f"pre-clear: tr.visible_content[:60]="
        f"{target_tr.visible_content[:60]!r} "
        f"state={target_tr.truncation_state}"
    );
    # Apply a hard_clear decision through the public helper so the
    # placeholder string matches what real pruning would write.
    apply_eviction_decision(EvictionDecision(
        tool_result=target_tr,
        kind="hard_clear"
    ));
    print(
        f"post-clear: tr.visible_content[:60]="
        f"{target_tr.visible_content[:60]!r} "
        f"state={target_tr.truncation_state}"
    );

    # Turn 2: another LLM call. materialize must fire again.
    follow_up("any updates?");
    count_after_t2 = policy.materialize_count;
    print(f"materialize calls after turn 2: {count_after_t2}");

    # Sweep the entire materialised list (any shape) for evidence of
    # the original planted value. A leak here would mean pruning failed
    # to propagate to LLM context.
    leaked = False;
    print(f"\nlast_materialized has {len(policy.last_materialized)} entries");
    for entry in policy.last_materialized {
        c = "";
        if isinstance(entry, Message) {
            c = str(getattr(entry, "content", ""));
        } elif isinstance(entry, dict) {
            c = str(entry.get("content", ""));
        }
        if "MAX_TOKEN_REFRESH_ATTEMPTS = 73" in c {
            leaked = True;
        }
    }

    failures = 0;

    if count_after_t1 >= 1 {
        print("\nPASS  materialize was called at least once during turn 1");
    } else {
        failures += 1;
        print("\nFAIL  materialize never fired during turn 1");
    }

    if count_after_t2 > count_after_t1 {
        print("PASS  materialize fired again during turn 2");
    } else {
        failures += 1;
        print("FAIL  materialize did not fire during turn 2");
    }

    if not leaked {
        print(
            "PASS  pruned value did NOT leak into the materialised LLM context"
        );
    } else {
        failures += 1;
        print(
            "FAIL  pruned tool result's original value reached the LLM context"
        );
    }

    if failures > 0 {
        raise AssertionError(f"{failures} assertion(s) failed");
    }
    print("\nAll assertions passed.");
}
