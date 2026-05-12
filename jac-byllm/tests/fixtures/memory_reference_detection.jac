"""Reference-detection smoke test.

Validates the post-Stage-3 fixes:

- After an assistant turn, `record_references_from_text` bumps
  `reference_count` and `last_referenced_at` on prior tool results whose
  identifiers appear in the assistant text.
- A `ReferencedBy` edge is created from the assistant MemoryMessage to
  each matched MemoryToolResult (citation topology, not just counters).
- A second turn that does NOT mention prior identifiers leaves the
  reference signal untouched.

Note: same-turn tool results are excluded from reference detection by
design (an assistant message can't cite a tool result that was produced
in the same turn — those results are what the model was responding to,
not cross-turn memory). The test therefore puts the citation in turn 2
against a tool result from turn 1.

Without the wiring this test exists to validate, the framework's
`ReferenceAwarePolicy` degrades to age-plus-size, silently erasing the
Direction 1 result on upstreamed code.
"""

import from byllm.lib { Model, MockToolCall }
import from byllm.memory {
    Memory,
    MemoryToolResult,
    MemoryMessage,
    ReferencedBy
}


def fetch_settings() -> str {
    # Distinctive identifiers the assistant can echo back. Matches the
    # snake/camel-≥6-chars and ALL_CAPS-≥4 rules in extract_identifiers.
    return (
        "LIGHTHOUSE_PROD_CONFIG\n"
        "MAX_TOKEN_REFRESH_ATTEMPTS = 73\n"
        "session_inactivity_minutes = 29\n"
    );
}


glob mem = Memory(persist=True);


glob llm = Model(
    model_name="mockllm",
    config={
        "outputs": [
            # Turn 1: tool call then a generic acknowledgement.
            # Importantly, the turn-1 final answer mentions NO identifiers
            # from fetch_settings — the same-turn citation guard would
            # skip them even if it did, so we save the actual citation
            # for turn 2.
            MockToolCall(tool=fetch_settings, args={}),
            "Looked it up.",
            # Turn 2: no tool call. Assistant echoes identifiers from
            # turn-1's tool result. This is the cross-turn citation that
            # record_references_from_text must detect.
            "Found MAX_TOKEN_REFRESH_ATTEMPTS — looks reasonable.",
            # Turn 3: control. Assistant says something unrelated; the
            # turn-1 tool result's reference signal must NOT advance.
            "All checks completed, nothing else to report.",
        ]
    },
);


def investigate(query: str) -> str by llm(
    tools=[fetch_settings], memory=mem
);


def follow_up(query: str) -> str by llm(memory=mem);


with entry {
    print("=== Reference detection smoke test ===\n");

    # ---- Turn 1: tool fires, assistant text has no identifiers ----
    investigate("what's in settings?");
    g = mem.graph;
    if g is None {
        raise AssertionError("Memory graph was not initialised");
    }

    settings_tr: MemoryToolResult | None = None;
    for tr in [g -->][?:MemoryToolResult] {
        if tr.tool_name == "fetch_settings" {
            settings_tr = tr;
        }
    }
    if settings_tr is None {
        raise AssertionError("expected a fetch_settings MemoryToolResult");
    }

    print(
        f"after turn 1: ref_count={settings_tr.reference_count} "
        f"last_referenced_at={settings_tr.last_referenced_at}"
    );

    # ---- Turn 2: assistant cites turn-1 identifier ----
    follow_up("anything notable?");
    print(
        f"after turn 2: ref_count={settings_tr.reference_count} "
        f"last_referenced_at={settings_tr.last_referenced_at}"
    );

    # Locate the assistant message that did the citing (turn == 2).
    turn2_asst: MemoryMessage | None = None;
    for m in [g -->][?:MemoryMessage] {
        if m.role == "assistant" and m.turn == 2 {
            turn2_asst = m;
        }
    }
    edge_targets_t2: list[MemoryToolResult] = [];
    if turn2_asst is not None {
        for tr in [turn2_asst ->:ReferencedBy:->][?:MemoryToolResult] {
            edge_targets_t2.append(tr);
        }
    }
    print(f"edges from turn-2 assistant: {len(edge_targets_t2)}");

    # ---- Turn 3: control. Same tool result must NOT accrue more refs ----
    ref_count_before_t3 = settings_tr.reference_count;
    last_ref_before_t3 = settings_tr.last_referenced_at;
    follow_up("anything else?");
    print(
        f"after turn 3: ref_count={settings_tr.reference_count} "
        f"last_referenced_at={settings_tr.last_referenced_at}"
    );

    # ---- Assertions ----
    failures = 0;

    if settings_tr.reference_count >= 1 {
        print("PASS  reference_count bumped on the cited tool result");
    } else {
        failures += 1;
        print(
            f"FAIL  reference_count not bumped "
            f"(got {settings_tr.reference_count})"
        );
    }

    if settings_tr.last_referenced_at > 0.0 {
        print("PASS  last_referenced_at populated");
    } else {
        failures += 1;
        print("FAIL  last_referenced_at left at zero");
    }

    if len(edge_targets_t2) >= 1 {
        print("PASS  ReferencedBy edge created from turn-2 assistant");
    } else {
        failures += 1;
        print("FAIL  no ReferencedBy edge from the turn-2 assistant message");
    }

    if settings_tr.reference_count == ref_count_before_t3 {
        print("PASS  unrelated turn 3 did not bump ref_count further");
    } else {
        failures += 1;
        print(
            f"FAIL  ref_count drifted on unrelated turn "
            f"({ref_count_before_t3} -> {settings_tr.reference_count})"
        );
    }

    if settings_tr.last_referenced_at == last_ref_before_t3 {
        print("PASS  unrelated turn 3 did not move last_referenced_at");
    } else {
        failures += 1;
        print("FAIL  last_referenced_at moved on an unrelated turn");
    }

    if failures > 0 {
        raise AssertionError(f"{failures} assertion(s) failed");
    }
    print("\nAll assertions passed.");
}
