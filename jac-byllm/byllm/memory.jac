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


"""Decision returned by `MemoryPolicy.select_eviction()`.

Each entry tells the pruner what to do with one MemoryToolResult node:
soft-trim (replace visible_content with head+tail), hard-clear (replace
with a placeholder), or skip. The kind is a string rather than an enum
to keep the surface stable across versions and make custom kinds
trivial for subclassing policies.
"""
obj EvictionDecision {
    has tool_result: MemoryToolResult,
        kind: str = "skip",       # skip | soft_trim | hard_clear
        head_chars: int = 1500,
        tail_chars: int = 1500,
        placeholder: str = "[Old tool result content cleared]";
}


"""Strategy interface for memory pruning, compaction, and materialization.

Concrete policies subclass this. The default methods are no-ops so a
caller-supplied policy can override only what it needs (e.g. only the
compaction step) without having to reimplement the rest. byLLM dispatches
into the policy at well-defined boundaries:

  - `should_prune`     : called pre-LLM-call. True triggers `select_eviction`.
  - `select_eviction`  : returns a list of EvictionDecisions to apply to
                         visible MemoryToolResults.
  - `should_compact`   : called pre-LLM-call after pruning. True triggers
                         `compact` and bypasses the framework's existing
                         token-threshold compaction path for this turn.
  - `compact`          : returns a MemoryCompactionSummary if a summary
                         was produced, else None. The policy is
                         expected to mutate the graph itself (hide
                         compacted nodes, attach the summary). Returning
                         None signals "no compaction was applied this
                         call" and the framework leaves the loop alone.
  - `materialize`      : returns the message list that should be sent to
                         the LLM this iteration. Default returns None
                         meaning "use mt_run.messages as-is".
"""
obj MemoryPolicy {
    def should_prune(graph: MemoryGraph, mt_run: object) -> bool;
    def select_eviction(
        graph: MemoryGraph, mt_run: object
    ) -> list[EvictionDecision];
    def should_compact(graph: MemoryGraph, mt_run: object) -> bool;
    def compact(
        graph: MemoryGraph, mt_run: object
    ) -> MemoryCompactionSummary | None;
    def materialize(graph: MemoryGraph, mt_run: object) -> list | None;
}


impl MemoryPolicy.should_prune(graph: MemoryGraph, mt_run: object) -> bool {
    return False;
}

impl MemoryPolicy.select_eviction(
    graph: MemoryGraph, mt_run: object
) -> list[EvictionDecision] {
    return [];
}

impl MemoryPolicy.should_compact(graph: MemoryGraph, mt_run: object) -> bool {
    return False;
}

impl MemoryPolicy.compact(
    graph: MemoryGraph, mt_run: object
) -> MemoryCompactionSummary | None {
    return None;
}

impl MemoryPolicy.materialize(
    graph: MemoryGraph, mt_run: object
) -> list | None {
    return None;
}


"""Age-based memory policy — the framework default.

Compaction-only for now. Fires `should_compact` when the visible
memory-resident token estimate (sum of MemoryMessage.token_estimate +
estimated tokens of visible MemoryToolResult.visible_content) crosses a
configurable threshold against `usable_budget_tokens`. Compaction picks
the oldest items beyond a `preserve_recent_turns` window and replaces
them with a synthetic placeholder summary.

This is a deliberately simple default. A future ReferenceAwarePolicy
(Stage 4) will subclass it and replace `select_eviction` /
`compact` with its score-based variants. A LLM-backed compactor will
override `compact` to call out to byLLM; this base version returns a
free-form text summary built from the eligible content's first lines.

Configurable parameters:
  - `usable_budget_tokens`     : the token budget the policy aims to stay under
  - `compact_threshold_ratio`  : fill fraction at or above which compaction fires
  - `preserve_recent_turns`    : how many recent turns are protected from compaction
"""
obj AgePolicy(MemoryPolicy) {
    has usable_budget_tokens: int = 8000,
        compact_threshold_ratio: float = 0.5,
        preserve_recent_turns: int = 2;

    override def should_compact(graph: MemoryGraph, mt_run: object) -> bool;
    override def compact(
        graph: MemoryGraph, mt_run: object
    ) -> MemoryCompactionSummary | None;
}


impl AgePolicy.should_compact(graph: MemoryGraph, mt_run: object) -> bool {
    if self.usable_budget_tokens <= 0 {
        return False;
    }
    total_tokens = 0;
    for m in [graph -->](?:MemoryMessage) {
        if m.visible {
            total_tokens += m.token_estimate;
        }
    }
    for tr in [graph -->](?:MemoryToolResult) {
        if tr.visible {
            # Approximate tokens at chars/4 — matches research-agent default.
            total_tokens += len(tr.visible_content) // 4;
        }
    }
    for s in [graph -->](?:MemoryCompactionSummary) {
        if s.visible {
            total_tokens += s.token_estimate;
        }
    }
    fill_ratio = total_tokens / float(self.usable_budget_tokens);
    return fill_ratio >= self.compact_threshold_ratio;
}


impl AgePolicy.compact(
    graph: MemoryGraph, mt_run: object
) -> MemoryCompactionSummary | None {
    cutoff = graph.turn_count - self.preserve_recent_turns;
    if cutoff <= 0 {
        return None;
    }
    msgs_to_compact = [
        m for m in [graph -->](?:MemoryMessage)
        if m.visible and m.turn <= cutoff
    ];
    trs_to_compact = [
        tr for tr in [graph -->](?:MemoryToolResult)
        if tr.visible and tr.produced_at_turn <= cutoff
    ];
    if not msgs_to_compact and not trs_to_compact {
        return None;
    }
    # Build a minimal free-form summary from the eligible content's first
    # lines. A future LLM-backed AgePolicy variant overrides this method.
    snippets = [];
    for m in msgs_to_compact {
        head = m.content.split("\n")[0][:120];
        if head {
            snippets.append(f"[{m.role} t{m.turn}] {head}");
        }
    }
    for tr in trs_to_compact {
        head = tr.full_content.split("\n")[0][:120];
        snippets.append(f"[tool {tr.tool_name} t{tr.produced_at_turn}] {head}");
    }
    summary_text = "Older context summary:\n" + "\n".join(snippets);

    turn_starts = [m.turn for m in msgs_to_compact] + [
        tr.produced_at_turn for tr in trs_to_compact
    ];
    covers_start = min(turn_starts) if turn_starts else 0;

    summary = MemoryCompactionSummary(
        generated_at=time.time(),
        covers_turn_start=covers_start,
        covers_turn_end=cutoff,
        summary_text=summary_text,
        dropped_message_count=len(msgs_to_compact),
        dropped_tool_result_count=len(trs_to_compact),
        token_estimate=len(summary_text) // 4
    );
    graph ++> summary;
    for m in msgs_to_compact {
        m.visible = False;
    }
    for tr in trs_to_compact {
        tr.visible = False;
    }
    return summary;
}


"""Public memory config. Pass to `by llm(memory=...)` to opt in.

Holds the live `MemoryGraph` after the first call so the caller can
inspect it. With `persist=True` the same graph is reused across
invocations of the decorated function (within the same process). With
`persist=False` a fresh graph is created on every invocation —
discarding the previous one.

Optional `policy` is the strategy interface for pruning/compaction/
materialization. When None (default for Stage 2), byLLM records into
the graph but doesn't act on it. When set, byLLM dispatches to the
policy's hooks at iteration boundaries, replacing the framework's
inline compaction code path.
"""
obj Memory {
    has persist: bool = True,
        session_id: str = "",
        policy: MemoryPolicy | None = None,
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
