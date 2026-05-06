"""Memory primitives for byLLM agents.

This module is the foundation of byLLM's built-in memory layer. The agent
author opts in by passing `memory=Memory(...)` to a `by llm()` declaration.
byLLM then auto-records tool results and messages into a per-agent
`MemoryGraph` and exposes the graph back to the caller for inspection.

Stage 2 deliberately stops at observability: the graph is populated, but
no pruning or compaction policy acts on it. Subsequent stages add the
`MemoryPolicy` interface and built-in policies (age-based,
reference-aware, typed-slot compaction) that operate on the same graph.

Layout:

  - `Memory`              — public config object passed at the call site.
                            Holds the live `MemoryGraph` after first call.
  - `MemoryGraph`         — node attached to `root()` (or standalone when
                            persist=False). Owns the per-session state.
  - `MemoryToolResult`    — one node per tool execution. Carries
                            full_content (never truncated) and
                            visible_content (what the LLM sees).
  - `MemoryMessage`       — one node per user/assistant/tool message.
  - `MemoryCompactionSummary` — placeholder for Stage 3+ summarisation.
  - `ReferencedBy`        — edge from MemoryMessage to MemoryToolResult,
                            written by Stage 5's citation channel.

ID scheme: `tr_<turn>_<tool_name>_<seq>` for tool results,
`msg_<turn>_<role>_<seq>` for messages. Stable within a session.
"""

import time;


node MemoryGraph {
    has session_id: str = "",
        created_at: float = 0.0,
        turn_count: int = 0,
        # Per-agent counters used to mint stable IDs without coordinating
        # across MTRuntime invocations.
        _tr_seq: int = 0,
        _msg_seq: int = 0,
        # Free-form stats bag for telemetry/analysis (token counts,
        # eviction history, etc.). Stage 3+ policies populate this.
        stats: dict = {};

    def mint_tool_result_id(turn: int, tool_name: str) -> str;
    def mint_message_id(turn: int, role: str) -> str;
    def record_tool_result(
        tool_name: str,
        args: dict,
        full_content: str,
        visible_content: str,
        turn: int,
        truncation_state: str = "full"
    ) -> object;
    def record_message(
        role: str,
        turn: int,
        content: str,
        token_estimate: int = 0
    ) -> object;
}


node MemoryToolResult {
    has tr_id: str,
        tool_name: str,
        full_content: str,
        args: dict = {},
        visible_content: str = "",
        produced_at: float = 0.0,
        produced_at_turn: int = 0,
        truncation_state: str = "full",
        visible: bool = True,
        reference_count: int = 0,
        last_referenced_at: float = 0.0;
}


node MemoryMessage {
    has msg_id: str,
        role: str,
        turn: int,
        content: str,
        token_estimate: int = 0,
        visible: bool = True;
}


node MemoryCompactionSummary {
    has generated_at: float = 0.0,
        covers_turn_start: int = 0,
        covers_turn_end: int = 0,
        summary_text: str = "",
        dropped_message_count: int = 0,
        dropped_tool_result_count: int = 0,
        token_estimate: int = 0,
        visible: bool = True;
}


edge ReferencedBy {
    has cite_turn: int = 0;
}


"""Public memory config. Pass to `by llm(memory=...)` to opt in.

Holds the live `MemoryGraph` after the first call so the caller can
inspect it. With `persist=True` the same graph is reused across
invocations of the decorated function (within the same process). With
`persist=False` a fresh graph is created on every invocation —
discarding the previous one.
"""
obj Memory {
    has persist: bool = True,
        session_id: str = "",
        # Set by byLLM at call time. Points at the live MemoryGraph node.
        graph: MemoryGraph | None = None;

    def get_or_create_graph() -> MemoryGraph;
    def reset() -> None;
}


impl MemoryGraph.mint_tool_result_id(turn: int, tool_name: str) -> str {
    self._tr_seq += 1;
    return f"tr_{turn}_{tool_name}_{self._tr_seq}";
}


impl MemoryGraph.mint_message_id(turn: int, role: str) -> str {
    self._msg_seq += 1;
    return f"msg_{turn}_{role}_{self._msg_seq}";
}


impl MemoryGraph.record_tool_result(
    tool_name: str,
    args: dict,
    full_content: str,
    visible_content: str,
    turn: int,
    truncation_state: str = "full"
) -> object {
    tr_id = self.mint_tool_result_id(turn, tool_name);
    tr = MemoryToolResult(
        tr_id=tr_id,
        tool_name=tool_name,
        args=args,
        full_content=full_content,
        visible_content=visible_content if visible_content else full_content,
        produced_at=time.time(),
        produced_at_turn=turn,
        truncation_state=truncation_state
    );
    self ++> tr;
    return tr;
}


impl MemoryGraph.record_message(
    role: str,
    turn: int,
    content: str,
    token_estimate: int = 0
) -> object {
    msg_id = self.mint_message_id(turn, role);
    msg = MemoryMessage(
        msg_id=msg_id,
        role=role,
        turn=turn,
        content=content,
        token_estimate=token_estimate
    );
    self ++> msg;
    return msg;
}


impl Memory.get_or_create_graph() -> MemoryGraph {
    if self.graph is not None and self.persist {
        return self.graph;
    }
    new_graph = MemoryGraph(
        session_id=self.session_id,
        created_at=time.time()
    );
    # Attach to the Jac root so the graph is part of the persistent
    # graph layer. Caller can also reach it via `mem.graph` directly.
    root() ++> new_graph;
    self.graph = new_graph;
    return new_graph;
}


impl Memory.reset() -> None {
    self.graph = None;
}
