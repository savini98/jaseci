"""Unit test for `AgePolicy.materialize` — graph-only, no LLM call.

Builds a ContextGraph by hand with a mix of:

- System messages on a mock mt_run.messages (graph doesn't store them)
- A visible compaction summary covering early turns
- A visible user message
- An assistant message with a tool_calls block (one tool call)
- A ContextToolResult paired by call_id whose visible_content was
  hard-cleared by pruning
- A hidden assistant message + matching hidden tool result (the case
  AgePolicy.compact would produce)

then calls `policy.materialize(graph, mock_mt_run)` and verifies the
returned list:

- Begins with system messages from mt_run
- Has the compaction summary in the right position
- Emits the visible assistant as a LiteLLM-shaped dict with tool_calls
- Emits the matching ToolCallResultMsg with the hard-cleared
  placeholder as content (the load-bearing behavior — pruning
  decisions actually reach the LLM context)
- Skips hidden messages and their tool results entirely
"""

import time;
import from jaclang.byllm.types { Message, MessageRole, ToolCallResultMsg }
import from jaclang.byllm.context {
    ContextGraph,
    ContextToolResult,
    ContextMessage,
    ContextCompactionSummary,
    AgePolicy
}


# Minimal stand-in for MTRuntime. materialize uses getattr(mt_run,
# "messages", ...) so any object exposing a `.messages` attribute works.
obj MockMtRun {
    has messages: list = [];
}


with entry {
    print("=== materialize unit test ===\n");

    graph = ContextGraph(session_id="t1", created_at=time.time());
    root() ++> graph;
    graph.turn_count = 4;

    # Visible: compaction summary that covers turns 0–1.
    graph ++> ContextCompactionSummary(
        generated_at=time.time(),
        covers_turn_start=0,
        covers_turn_end=1,
        summary_text="Earlier turns recap.",
        dropped_message_count=2,
        dropped_tool_result_count=1,
        visible=True
    );

    # Hidden: an old assistant message + tool result. AgePolicy.compact
    # would hide pairs together — materialize must skip both.
    graph.record_message(role="user", turn=1, content="hidden user");
    hidden_assistant = graph.record_message(
        role="assistant",
        turn=1,
        content="hidden assistant",
        tool_calls=[{"id": "ghost_call_x", "name": "ghost", "args": {}}]
    );
    hidden_assistant.visible = False;
    graph.record_tool_result(
        tool_name="ghost",
        args={},
        full_content="should not appear",
        visible_content="should not appear",
        turn=1,
        call_id="ghost_call_x"
    );
    # Hide the hidden tool result manually (matches the pair).
    for tr in [graph -->][?:ContextToolResult] {
        if tr.call_id == "ghost_call_x" {
            tr.visible = False;
        }
    }

    # Visible turn 2: user msg
    graph.record_message(role="user", turn=2, content="check the config");

    # Visible turn 3: assistant with tool_calls + matching tool result
    # whose visible_content was hard-cleared by pruning.
    graph.record_message(
        role="assistant",
        turn=3,
        content="opening prod.yaml",
        tool_calls=[
            {"id": "call_prod", "name": "read_file", "args": {"path": "prod.yaml"}}
        ]
    );
    tr_prod = graph.record_tool_result(
        tool_name="read_file",
        args={"path": "prod.yaml"},
        full_content="keepalive_seconds: 47\nbind_port: 8080\n",
        visible_content="keepalive_seconds: 47\nbind_port: 8080\n",
        turn=3,
        call_id="call_prod"
    );
    # Simulate pruning hard-clearing T3's tool result.
    tr_prod.visible_content = "[Old tool result content cleared]";
    tr_prod.truncation_state = "hard_cleared";

    # Visible turn 4: plain assistant message, no tool calls.
    graph.record_message(role="assistant", turn=4, content="done.");

    # Mock mt_run with one SYSTEM message.
    mt = MockMtRun(messages=[
        Message(role=MessageRole.SYSTEM, content="you are a helpful assistant")
    ]);

    policy = AgePolicy(
        usable_budget_tokens=1000,
        compact_threshold_ratio=0.5,
        preserve_recent_turns=1
    );
    out = policy.materialize(graph, mt);
    if out is None {
        raise AssertionError("materialize returned None");
    }
    print(f"materialize produced {len(out)} entries:\n");
    for entry in out {
        if isinstance(entry, Message) {
            etype = type(entry).__name__;
            erole = getattr(entry.role, "value", entry.role);
            econt = str(getattr(entry, "content", ""))[:80];
            print(f"  [{etype}] role={erole} content={econt!r}");
        } elif isinstance(entry, dict) {
            r = entry.get("role", "?");
            c = str(entry.get("content", ""))[:80];
            tcs = entry.get("tool_calls", []);
            print(f"  [dict] role={r} content={c!r} tool_calls={len(tcs)}");
        } else {
            print(f"  [{type(entry).__name__}] {entry!r}");
        }
    }

    # ---- Assertions ----
    failures = 0;

    # 1. System message comes first.
    if (
        len(out) > 0
        and isinstance(out[0], Message)
        and out[0].role == MessageRole.SYSTEM
    ) {
        print("\nPASS  system message emitted first");
    } else {
        failures += 1;
        print("\nFAIL  system message not at the front of output");
    }

    # 2. Compaction summary present as a user-role marker.
    summary_text_present = False;
    for entry in out {
        if (
            isinstance(entry, Message)
            and entry.role == MessageRole.USER
            and "Earlier turns recap." in str(getattr(entry, "content", ""))
        ) {
            summary_text_present = True;
        }
    }
    if summary_text_present {
        print("PASS  compaction summary emitted inline");
    } else {
        failures += 1;
        print("FAIL  compaction summary missing from output");
    }

    # 3. The hidden ghost_call_x does NOT appear anywhere.
    ghost_present = False;
    for entry in out {
        # Inspect dict entries' tool_calls and ToolCallResultMsg call_ids.
        if isinstance(entry, dict) {
            for tc in (entry.get("tool_calls") or []) {
                if isinstance(tc, dict) and tc.get("id") == "ghost_call_x" {
                    ghost_present = True;
                }
            }
        } elif isinstance(entry, ToolCallResultMsg) {
            if getattr(entry, "tool_call_id", "") == "ghost_call_x" {
                ghost_present = True;
            }
        }
    }
    if not ghost_present {
        print("PASS  hidden turn (ghost_call_x) skipped entirely");
    } else {
        failures += 1;
        print("FAIL  hidden ghost_call_x leaked into materialised output");
    }

    # 4. Assistant message for turn-3 emitted as a dict with the
    #    call_prod tool_use block.
    assistant_call_emitted = False;
    for entry in out {
        if (
            isinstance(entry, dict)
            and entry.get("role") == "assistant"
        ) {
            for tc in (entry.get("tool_calls") or []) {
                if isinstance(tc, dict) and tc.get("id") == "call_prod" {
                    assistant_call_emitted = True;
                }
            }
        }
    }
    if assistant_call_emitted {
        print("PASS  assistant turn 3 emitted with call_prod tool_use");
    } else {
        failures += 1;
        print("FAIL  assistant turn 3 tool_use block missing");
    }

    # 5. The matching ToolCallResultMsg for call_prod has the HARD-
    #    CLEARED placeholder as content, not the original
    #    keepalive_seconds value. This is the R2-fix invariant.
    cleared_tr_emitted = False;
    leaked_value = False;
    for entry in out {
        if (
            isinstance(entry, ToolCallResultMsg)
            and getattr(entry, "tool_call_id", "") == "call_prod"
        ) {
            content_s = str(getattr(entry, "content", ""));
            if "[Old tool result content cleared]" in content_s {
                cleared_tr_emitted = True;
            }
            if "keepalive_seconds: 47" in content_s {
                leaked_value = True;
            }
        }
    }
    if cleared_tr_emitted and not leaked_value {
        print(
            "PASS  pruned tool result emits placeholder (LLM no longer sees the value)"
        );
    } else {
        failures += 1;
        if not cleared_tr_emitted {
            print("FAIL  no matching ToolCallResultMsg for call_prod");
        }
        if leaked_value {
            print(
                "FAIL  pruned tool result leaked the original value into materialised output"
            );
        }
    }

    # 6. The plain-text turn-4 assistant message appears as a Message
    #    (no tool_calls).
    plain_t4 = False;
    for entry in out {
        if (
            isinstance(entry, Message)
            and getattr(entry, "role", None) == MessageRole.ASSISTANT
            and "done." in str(getattr(entry, "content", ""))
        ) {
            plain_t4 = True;
        }
    }
    if plain_t4 {
        print("PASS  plain assistant turn 4 emitted as a Message");
    } else {
        failures += 1;
        print("FAIL  plain assistant turn 4 missing");
    }

    if failures > 0 {
        raise AssertionError(f"{failures} assertion(s) failed");
    }
    print("\nAll assertions passed.");
}
