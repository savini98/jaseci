"""Stage 2 smoke test: memory=Memory() populates a MemoryGraph that's
reachable from the caller after the LLM call.

Validates:
- After the call, mem.graph is a populated MemoryGraph node
- The user message and assistant message are recorded as MemoryMessage nodes
- Tool calls are recorded as MemoryToolResult nodes with stable tr_ids
- persist=False produces a fresh graph on the second call
- persist=True keeps the graph alive across calls
"""

import from byllm.lib { Model, MockToolCall }
import from byllm.memory { Memory, MemoryGraph, MemoryToolResult, MemoryMessage }


def fetch_status() -> str {
    return "service-ok";
}


glob llm = Model(
    model_name="mockllm",
    config={
        "outputs": [
            # Call 1: tool call then plain-string final
            MockToolCall(tool=fetch_status, args={}),
            "service-ok",
            # Call 2 (persist test): just a plain final
            "service-ok again",
            # Call 3 (persist=False fresh-graph test): plain final
            "fresh response",
        ]
    },
);


glob persistent_mem = Memory(persist=True);
glob fresh_mem = Memory(persist=False);


def check_status_persistent(query: str) -> str by llm(
    tools=[fetch_status], memory=persistent_mem
);

def check_status_fresh(query: str) -> str by llm(
    memory=fresh_mem
);


with entry {
    print("=== Stage 2: memory=Memory() populates MemoryGraph ===\n");

    # ---- First persistent call: tool fires + final answer ----
    r1 = check_status_persistent("status?");
    print(f"call 1 result: {r1!r}");
    g1 = persistent_mem.graph;
    print(f"call 1 graph turn_count={g1.turn_count}");

    # Walk the graph for tool results and messages.
    tool_results_1 = [tr for tr in [g1 -->](?:MemoryToolResult)];
    messages_1 = [m for m in [g1 -->](?:MemoryMessage)];
    tool_ids_1 = [tr.tr_id for tr in tool_results_1];
    print(f"call 1 tool_results: {tool_ids_1}");
    print(f"call 1 message count: {len(messages_1)}");

    # ---- Second persistent call: no tool, but graph should be reused ----
    r2 = check_status_persistent("again");
    g2 = persistent_mem.graph;
    print(f"\ncall 2 result: {r2!r}");
    print(f"call 2 graph is same object: {g2 is g1}");
    print(f"call 2 graph turn_count={g2.turn_count}");
    messages_2 = [m for m in [g2 -->](?:MemoryMessage)];
    print(f"call 2 cumulative message count: {len(messages_2)}");

    # ---- Fresh-mode call: separate graph, NOT same node ----
    r3 = check_status_fresh("fresh");
    g3 = fresh_mem.graph;
    print(f"\ncall 3 (fresh) result: {r3!r}");
    print(f"call 3 graph is distinct from persistent: {g3 is not g1}");

    # ---- Assertions ----
    failures = 0;
    if g1 is not None and g2 is not None and g1 is g2 {
        print("\nPASS  persist=True reuses the same graph node");
    } else {
        failures += 1;
        print("FAIL  persist=True did not preserve graph identity");
    }
    if len(tool_results_1) >= 1 {
        print("PASS  tool result recorded as MemoryToolResult");
    } else {
        failures += 1;
        print(f"FAIL  expected ≥1 tool result, got {len(tool_results_1)}");
    }
    if any(tr.tool_name == "fetch_status" for tr in tool_results_1) {
        print("PASS  recorded tool_name == 'fetch_status'");
    } else {
        failures += 1;
        print(f"FAIL  no tool_result with tool_name 'fetch_status' found");
    }
    if any(tr.tr_id.startswith("tr_") and "fetch_status" in tr.tr_id for tr in tool_results_1) {
        print("PASS  tr_id follows the documented scheme");
    } else {
        failures += 1;
        print(f"FAIL  tr_id scheme mismatch: {tool_ids_1}");
    }
    msg_roles_1 = [m.role for m in messages_1];
    if "user" in msg_roles_1 {
        print("PASS  user message recorded on call 1");
    } else {
        failures += 1;
        print(f"FAIL  no user message recorded. roles={msg_roles_1}");
    }
    if "assistant" in msg_roles_1 {
        print("PASS  assistant message recorded on call 1");
    } else {
        failures += 1;
        print(f"FAIL  no assistant message recorded. roles={msg_roles_1}");
    }
    if len(messages_2) > len(messages_1) {
        print("PASS  call 2 added more messages to the persisted graph");
    } else {
        failures += 1;
        print(f"FAIL  call 2 didn't grow message count: {len(messages_1)} → {len(messages_2)}");
    }
    if g3 is not None and g3 is not g1 {
        print("PASS  persist=False yielded a separate graph node");
    } else {
        failures += 1;
        print("FAIL  persist=False reused the persistent graph (or no graph)");
    }

    if failures > 0 {
        raise AssertionError(f"{failures} assertion(s) failed");
    }
    print("\nAll assertions passed.");
}
