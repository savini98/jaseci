"""Context primitives for byLLM agents.

This module is the foundation of byLLM's built-in context management
layer. The agent author opts in by passing `context=Context(...)` to a
`by llm()` call. byLLM then auto-records tool results and messages into a
per-agent `ContextGraph` and exposes the graph back to the caller for
inspection. An optional `ContextPolicy` (age-based, reference-aware, or
caller-defined) acts on that graph at iteration boundaries: pruning old
tool results, compacting old turns into summaries, and rebuilding the
LLM message list from the graph.

Design:
  - `Context`              — public config object passed at the call site.
                             Holds the live `ContextGraph` after first call.
  - `ContextGraph`         — node attached to `root()`. Owns the
                             per-session state.
  - `ContextToolResult`    — one node per tool execution. Carries
                             full_content (never truncated) and
                             visible_content (what the LLM sees).
  - `ContextMessage`       — one node per user/assistant/tool message.
  - `ContextCompactionSummary` — one node per compaction pass; replaces
                             the turns it covers in the materialized list.
  - `ReferencedBy`         — edge from ContextMessage to ContextToolResult,
                             written by reference detection when an
                             assistant reply cites a tool result.

ID scheme: `tr_<turn>_<tool_name>_<seq>` for tool results,
`msg_<turn>_<role>_<seq>` for messages. Stable within a session.
"""

import time;
import re;
import json;
import from jaclang.byllm.types { Message, MessageRole, ToolCallResultMsg }


node ContextGraph {
    has session_id: str = "",
        created_at: float = 0.0,
        turn_count: int = 0,
        # Per-agent counters used to mint stable IDs without coordinating
        # across MTRuntime invocations.
        _tr_seq: int = 0,
        _msg_seq: int = 0,
        # Free-form stats bag for telemetry/analysis (token counts,
        # eviction history, etc.). Policies populate this.
        stats: dict = {};

    def mint_tool_result_id(turn: int, tool_name: str) -> str;
    def mint_message_id(turn: int, role: str) -> str;
    def record_tool_result(
        tool_name: str,
        args: dict,
        full_content: str,
        visible_content: str,
        turn: int,
        truncation_state: str = "full",
        call_id: str = ""
    ) -> object;

    def record_message(
        role: str,
        turn: int,
        content: str,
        token_estimate: int = 0,
        tool_calls: list | None = None
    ) -> object;
}


node ContextToolResult {
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
        last_referenced_at: float = 0.0,
        # tool_use_id from the originating assistant message's
        # tool_calls block. Empty when the caller didn't supply one.
        # Materialize uses this to pair tool_result blocks back to
        # their tool_use block when rebuilding the LLM message list.
        call_id: str = "";
}


node ContextMessage {
    has msg_id: str,
        role: str,
        turn: int,
        content: str,
        token_estimate: int = 0,
        visible: bool = True,
        # For assistant messages that emit tool_use blocks: the
        # structured tool_calls list (each entry: {id, name, args}).
        # Empty for non-assistant turns or assistant turns with no
        # tools. Lets materialize rebuild the tool_use blocks back
        # into the LLM message list so the API-required
        # tool_use_id ↔ tool_result pairing survives pruning.
        tool_calls: list = [];
}


node ContextCompactionSummary {
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


"""Apply a single eviction decision to its target ToolResult.

Returns the number of tokens (chars/4 estimate) freed from the
visible_content. Idempotent: calling on an already-cleared node is a
no-op and returns 0. Stays a free function (not a ContextPolicy
method) so the loop can apply decisions returned from any policy
without dispatching through that policy again.
"""
def apply_eviction_decision(decision: object) -> int {
    tr = getattr(decision, "tool_result", None);
    if tr is None {
        return 0;
    }
    kind = getattr(decision, "kind", "skip");
    if kind == "skip" {
        return 0;
    }
    if getattr(tr, "truncation_state", "") == "hard_cleared" {
        return 0;
    }
    before = len(getattr(tr, "visible_content", "")) // 4;
    if kind == "hard_clear" {
        tr.visible_content = getattr(
            decision, "placeholder", "[Old tool result content cleared]"
        );
        tr.truncation_state = "hard_cleared";
    } elif kind == "soft_trim" {
        head = int(getattr(decision, "head_chars", 1500));
        tail = int(getattr(decision, "tail_chars", 1500));
        cur: str = str(tr.visible_content);
        if len(cur) > head + tail {
            tr.visible_content = cur[:head] + "\n...[soft-trimmed]...\n" + cur[-tail:];
            if tr.truncation_state == "full" {
                tr.truncation_state = "head_tail";
            }
        }
    }
    after = len(tr.visible_content) // 4;
    return max(0, before - after);
}


"""Decision returned by `ContextPolicy.select_eviction()`.

Each entry tells the pruner what to do with one ContextToolResult node:
soft-trim (replace visible_content with head+tail), hard-clear (replace
with a placeholder), or skip. The kind is a string rather than an enum
to keep the surface stable across versions and make custom kinds
trivial for subclassing policies.
"""
obj EvictionDecision {
    has tool_result: ContextToolResult,
        kind: str = "skip",  # skip | soft_trim | hard_clear
        head_chars: int = 1500,
        tail_chars: int = 1500,
        placeholder: str = "[Old tool result content cleared]";
}


"""Strategy interface for context pruning, compaction, and materialization.

Concrete policies subclass this. The default methods are no-ops so a
caller-supplied policy can override only what it needs (e.g. only the
compaction step) without having to reimplement the rest. byLLM dispatches
into the policy at well-defined boundaries:

  - `should_prune`     : called pre-LLM-call. True triggers `select_eviction`.
  - `select_eviction`  : returns a list of EvictionDecisions to apply to
                         visible ContextToolResults.
  - `should_compact`   : called pre-LLM-call after pruning. True triggers
                         `compact` and bypasses the framework's existing
                         token-threshold compaction path for this turn.
  - `compact`          : returns a ContextCompactionSummary if a summary
                         was produced, else None. The policy is
                         expected to mutate the graph itself (hide
                         compacted nodes, attach the summary). Returning
                         None signals "no compaction was applied this
                         call" and the framework leaves the loop alone.
  - `materialize`      : returns the message list that should be sent to
                         the LLM this iteration. Default returns None
                         meaning "use mt_run.messages as-is".
"""
obj ContextPolicy {
    def should_prune(graph: ContextGraph, mt_run: object) -> bool;
    def select_eviction(graph: ContextGraph, mt_run: object) -> list[EvictionDecision];
    def should_compact(graph: ContextGraph, mt_run: object) -> bool;
    def compact(
        graph: ContextGraph, mt_run: object
    ) -> ContextCompactionSummary | None;

    def materialize(graph: ContextGraph, mt_run: object) -> list | None;
}


impl ContextPolicy.should_prune(graph: ContextGraph, mt_run: object) -> bool {
    return False;
}

impl ContextPolicy.select_eviction(
    graph: ContextGraph, mt_run: object
) -> list[EvictionDecision] {
    return [];
}

impl ContextPolicy.should_compact(graph: ContextGraph, mt_run: object) -> bool {
    return False;
}

impl ContextPolicy.compact(
    graph: ContextGraph, mt_run: object
) -> ContextCompactionSummary | None {
    return None;
}

impl ContextPolicy.materialize(graph: ContextGraph, mt_run: object) -> list | None {
    return None;
}


"""Age-based context policy — the simplest built-in.

Compaction-only: `should_compact` fires when the visible token estimate
(sum of ContextMessage.token_estimate + estimated tokens of visible
ContextToolResult.visible_content) crosses a configurable threshold
against `usable_budget_tokens`. Compaction picks the oldest items beyond
a `preserve_recent_turns` window and replaces them with a synthetic
placeholder summary built from the eligible content's first lines.

Serves as the base for richer policies: `ReferenceAwarePolicy` (below)
adds score-based eviction, and `TypedSlotCompactionPolicy`
(typed_compaction.jac) replaces the free-form summary with an LLM-backed
structured extraction.

Configurable parameters:
  - `usable_budget_tokens`     : the token budget the policy aims to stay under
  - `compact_threshold_ratio`  : fill fraction at or above which compaction fires
  - `preserve_recent_turns`    : how many recent turns are protected from compaction
"""
obj AgePolicy(ContextPolicy) {
    has usable_budget_tokens: int = 8000,
        compact_threshold_ratio: float = 0.5,
        preserve_recent_turns: int = 2;

    override def should_compact(graph: ContextGraph, mt_run: object) -> bool;
    override def compact(
        graph: ContextGraph, mt_run: object
    ) -> ContextCompactionSummary | None;

    override def materialize(graph: ContextGraph, mt_run: object) -> list | None;
}


impl AgePolicy.should_compact(graph: ContextGraph, mt_run: object) -> bool {
    if self.usable_budget_tokens <= 0 {
        return False;
    }
    total_tokens = 0;
    for m in [graph-->][?:ContextMessage] {
        if m.visible {
            total_tokens += m.token_estimate;
        }
    }
    for tr in [graph-->][?:ContextToolResult] {
        if tr.visible {
            # Approximate tokens at chars/4.
            total_tokens += len(tr.visible_content) // 4;
        }
    }
    for s in [graph-->][?:ContextCompactionSummary] {
        if s.visible {
            total_tokens += s.token_estimate;
        }
    }
    fill_ratio = total_tokens / float(self.usable_budget_tokens);
    return fill_ratio >= self.compact_threshold_ratio;
}


impl AgePolicy.compact(
    graph: ContextGraph, mt_run: object
) -> ContextCompactionSummary | None {
    cutoff = graph.turn_count - self.preserve_recent_turns;
    if cutoff <= 0 {
        return None;
    }
    msgs_to_compact = [
        m
        for m in [graph-->][?:ContextMessage]
        if m.visible and m.turn <= cutoff
    ];
    trs_to_compact = [
        tr
        for tr in [graph-->][?:ContextToolResult]
        if tr.visible and tr.produced_at_turn <= cutoff
    ];
    if not msgs_to_compact and not trs_to_compact {
        return None;
    }
    # Build a minimal free-form summary from the eligible content's first
    # lines. LLM-backed variants (e.g. TypedSlotCompactionPolicy) override
    # this method.
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

    summary = ContextCompactionSummary(
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


"""Rebuild the LLM message list from the graph.

Called by `MTRuntime.dispatch_params` just before each LLM call when a
ContextPolicy is wired. The returned list REPLACES `mt_run.messages`
for that one call; `mt_run.messages` itself is never mutated.

Layout:

1. System messages from `mt_run.messages` (we don't put system msgs
   on the graph).
2. Visible ContextCompactionSummary nodes inline, ordered by
   `covers_turn_end`. Each summary becomes a user-role marker message
   in place of the original turns it covered.
3. Visible ContextMessage nodes in turn order. Plain user / assistant
   messages emit as Message objects. Assistant messages that carried
   `tool_calls` emit as LiteLLM-shaped dicts (Message has no
   `tool_calls` field) so the wire-format `tool_use` blocks survive.
   Each tool_call entry is followed by its matching ContextToolResult
   as a ToolCallResultMsg — paired by `call_id` so the Anthropic API
   contract (every tool_use must have a matching tool_result) holds.

Invariants this design relies on:

- `apply_eviction_decision` never sets `visible=False`. It only
  mutates `visible_content` (placeholder for hard_clear, head+tail
  for soft_trim). So pruning preserves pair-matching: the tool_result
  is still emitted, just with placeholder content.
- `AgePolicy.compact` hides assistant messages and their tool results
  together (both selected by `turn <= cutoff`). So compaction also
  preserves pair-matching: both absent.

This is why no "drop orphaned tool_use" logic is needed — the graph's
own consistency guarantees the materialized list is valid.
"""
impl AgePolicy.materialize(graph: ContextGraph, mt_run: object) -> list | None {
    out: list = [];

    # 1. System messages from mt_run.messages — these live outside the
    # graph (the factory's bulk-record loop copies their content into
    # ContextMessage too, but we re-emit from the source here so the
    # wire-format role is exactly what byLLM constructed).
    # `mt_run` is typed as `object` on the policy interface so concrete
    # policies stay decoupled from MTRuntime — use getattr to read the
    # messages list defensively.
    raw_msgs = getattr(mt_run, "messages", None) or [];
    for m in raw_msgs {
        if isinstance(m, Message) and m.role == MessageRole.SYSTEM {
            out.append(m);
        }
    }

    # 2. Visible compaction summaries inline.
    summaries: list = [
        s
        for s in [graph-->][?:ContextCompactionSummary]
        if s.visible
    ];
    summaries.sort(key=lambda (s: ContextCompactionSummary) { s.covers_turn_end; });
    for s in summaries {
        out.append(
            Message(
                role=MessageRole.USER,
                content=(
                    f"[Summary of turns {s.covers_turn_start}-"
                    f"{s.covers_turn_end}]\n{s.summary_text}"
                )
            )
        );
    }

    # 3. Build a fast lookup from call_id → ContextToolResult so we can
    # pair each tool_use block back to its result in O(1) instead of
    # scanning the graph per tool call.
    tr_by_call_id: dict = {};
    for tr in [graph-->][?:ContextToolResult] {
        if tr.visible and tr.call_id {
            tr_by_call_id[tr.call_id] = tr;
        }
    }

    # 4. Visible ContextMessages in turn order, with their tool results
    # spliced in immediately after each assistant tool_calls turn.
    msgs: list = [
        m
        for m in [graph-->][?:ContextMessage]
        if m.visible
    ];
    msgs.sort(key=lambda (m: ContextMessage) { m.turn; });
    for m in msgs {
        if m.role == "system" {
            # Already emitted from mt_run above; skip.
            continue;
        } elif m.role == "user" {
            out.append(Message(role=MessageRole.USER, content=m.content));
        } elif m.role == "assistant" {
            if m.tool_calls {
                # Emit assistant message WITH tool_use blocks as a
                # LiteLLM-shaped dict (byLLM Message has no tool_calls
                # field). MTRuntime.get_msg_list passes dicts through
                # without to_dict, so this lands on the wire intact.
                # Each tool_calls entry was normalised by
                # _extract_tool_calls in mtir.impl.jac into a dict of
                # shape {id, name, args}. We narrow with isinstance so
                # the type checker is happy with the .get() calls.
                tcs_serialized: list = [];
                for tc in m.tool_calls {
                    if not isinstance(tc, dict) {
                        continue;
                    }
                    tc_id_s: str = str(tc.get("id", ""));
                    tc_name_s: str = str(tc.get("name", ""));
                    tc_args_obj = tc.get("args", {});
                    tcs_serialized.append(
                        {
                            "id": tc_id_s,
                            "type": "function",
                            "function": {
                                "name": tc_name_s,
                                "arguments": json.dumps(tc_args_obj)
                            }
                        }
                    );
                }
                out.append(
                    {
                        "role": "assistant",
                        "content": str(m.content),
                        "tool_calls": tcs_serialized
                    }
                );
                # Emit each matching tool result. visible_content
                # carries whatever pruning decided — placeholder if
                # hard_cleared, head+tail if soft_trimmed, full
                # otherwise.
                for tc in m.tool_calls {
                    if not isinstance(tc, dict) {
                        continue;
                    }
                    tc_id_s2: str = str(tc.get("id", ""));
                    tc_name_s2: str = str(tc.get("name", "")) or "unknown";
                    tr = tr_by_call_id.get(tc_id_s2);
                    if tr is not None {
                        out.append(
                            ToolCallResultMsg(
                                content=str(tr.visible_content),
                                tool_call_id=tc_id_s2,
                                name=str(tr.tool_name)
                            )
                        );
                    } else {
                        # Defensive stub: keeps tool_use_id matched
                        # when no ContextToolResult was recorded.
                        # Shouldn't happen in practice.
                        out.append(
                            ToolCallResultMsg(
                                content="[tool result not recorded]",
                                tool_call_id=tc_id_s2,
                                name=tc_name_s2
                            )
                        );
                    }
                }
            } else {
                # Plain assistant message with no tool calls.
                out.append(Message(role=MessageRole.ASSISTANT, content=m.content));
            }
        }
    }

    return out;
}


"""Pull "distinctive" tokens out of tool-result text for substring-based
reference detection. Two patterns chosen to over-index on content
identifiers and under-index on prose: snake/camel identifiers ≥ 6 chars
and ALL-CAPS tokens. Path-like matches (containing a slash) are
deliberately excluded — generic project paths appear in many tool
outputs and inflate reference counts spuriously.
"""
def extract_identifiers(text: str, min_len: int = 6) -> list[str] {
    idents = re.findall(r"[A-Za-z_][A-Za-z0-9_]{" + str(min_len - 1) + r",}", text);
    caps = re.findall(r"\b[A-Z][A-Z0-9_]{3,}\b", text);
    seen: dict = {};
    out: list[str] = [];
    for x in idents + caps {
        if x not in seen {
            seen[x] = True;
            out.append(x);
        }
    }
    return out;
}


"""Update reference signals on the given graph after an assistant
message is produced. Walks visible ContextToolResult nodes from earlier
turns and increments `reference_count` + `last_referenced_at` whenever
any of the result's distinctive identifiers appear as substrings in
the new assistant text. When the corresponding assistant
ContextMessage at `current_turn` is found on the graph, also creates a
`ReferencedBy(cite_turn=current_turn)` edge from it to each matched
ContextToolResult — letting downstream walkers see the actual citation
topology, not just per-node counters. Returns the count of tool results
that got credit this call.

This is a free function (rather than a method on ContextPolicy) so the
loop can call it once per turn regardless of whether a policy is
attached — reference signal is useful purely for telemetry too.
"""
def record_references_from_text(
    graph: ContextGraph, assistant_text: str, current_turn: int
) -> int {
    if not assistant_text {
        return 0;
    }
    # Locate the assistant ContextMessage we'll attach citation edges from.
    # The append-only graph means "the most recent assistant message at
    # current_turn" is the one record_message just produced.
    cite_source: ContextMessage | None = None;
    for m in [graph-->][?:ContextMessage] {
        if m.role == "assistant" and m.turn == current_turn {
            cite_source = m;
        }
    }

    detections = 0;
    for tr in [graph-->][?:ContextToolResult] {
        if tr.produced_at_turn >= current_turn {
            continue;
        }
        idents = extract_identifiers(tr.full_content);
        hit = False;
        for ident in idents {
            if len(ident) >= 6 and ident in assistant_text {
                hit = True;
                break;
            }
        }
        if hit {
            tr.reference_count += 1;
            tr.last_referenced_at = time.time();
            detections += 1;
            if cite_source is not None {
                cite_source +>:ReferencedBy(cite_turn=current_turn):+> tr;
            }
        }
    }
    return detections;
}


"""Reference-aware context policy.

Inherits AgePolicy's compaction behavior. Overrides `should_prune` and
`select_eviction` to score visible ContextToolResults on
(reference_count, recency, age, size) and pick eviction order from
lowest-score upward: content the assistant has actually cited is
preserved through pressure while unreferenced results are evicted
first. Stops evicting as soon as the visible tool-result fill drops to
`target_ratio * usable_budget_tokens`.
"""
obj ReferenceAwarePolicy(AgePolicy) {
    has weight_ref_count: float = 10.0,
        weight_recency: float = 1.0,
        weight_age: float = -1.0,
        weight_size: float = -0.5,
        soft_threshold_ratio: float = 0.3,
        hard_threshold_ratio: float = 0.5,
        target_ratio: float = 0.5,
        keep_last_turns: int = 1,
        soft_head_chars: int = 1500,
        soft_tail_chars: int = 1500;

    override def should_prune(graph: ContextGraph, mt_run: object) -> bool;
    override def select_eviction(
        graph: ContextGraph, mt_run: object
    ) -> list[EvictionDecision];

    def _score(tr: ContextToolResult, current_turn: int) -> float;
}


impl ReferenceAwarePolicy.should_prune(graph: ContextGraph, mt_run: object) -> bool {
    if self.usable_budget_tokens <= 0 {
        return False;
    }
    total_tokens = 0;
    for m in [graph-->][?:ContextMessage] {
        if m.visible {
            total_tokens += m.token_estimate;
        }
    }
    for tr in [graph-->][?:ContextToolResult] {
        if tr.visible {
            total_tokens += len(tr.visible_content) // 4;
        }
    }
    fill_ratio = total_tokens / float(self.usable_budget_tokens);
    return fill_ratio >= self.soft_threshold_ratio;
}


impl ReferenceAwarePolicy._score(tr: ContextToolResult, current_turn: int) -> float {
    age = float(current_turn - tr.produced_at_turn);
    size_tokens = float(len(tr.visible_content) // 4);
    recency = 1.0 if tr.last_referenced_at > 0.0 else 0.0;
    return (
        self.weight_ref_count * float(tr.reference_count) + self.weight_recency * recency + self.weight_age * age + self.weight_size * (
            size_tokens / 1000.0
        )
    );
}


impl ReferenceAwarePolicy.select_eviction(
    graph: ContextGraph, mt_run: object
) -> list[EvictionDecision] {
    keep_floor = graph.turn_count - self.keep_last_turns;
    candidates = [
        tr
        for tr in [graph-->][?:ContextToolResult]
        if tr.visible
        and tr.truncation_state != "hard_cleared"
        and tr.produced_at_turn <= keep_floor
    ];
    current_turn = graph.turn_count;
    candidates.sort(key=lambda (tr: ContextToolResult) { self._score(tr, current_turn); });

    # Compute starting tool-result token total and evict-mode threshold.
    running_tool_tokens = 0;
    total_tokens = 0;
    for tr_all in [graph-->][?:ContextToolResult] {
        if tr_all.visible {
            tk = len(tr_all.visible_content) // 4;
            running_tool_tokens += tk;
            total_tokens += tk;
        }
    }
    for m in [graph-->][?:ContextMessage] {
        if m.visible {
            total_tokens += m.token_estimate;
        }
    }
    fill_ratio = (
        total_tokens / float(self.usable_budget_tokens)
            if self.usable_budget_tokens > 0
            else 0.0
    );
    hard_mode = fill_ratio >= self.hard_threshold_ratio;
    target = int(self.target_ratio * float(self.usable_budget_tokens));

    decisions: list[EvictionDecision] = [];
    for tr in candidates {
        # Early-stop: stop evicting once tool-result fill is at or below
        # target. `<=` (not `<`) so hitting the target exactly does not
        # evict one extra candidate.
        if running_tool_tokens <= target {
            break;
        }
        before = len(tr.visible_content) // 4;
        kind = "hard_clear" if hard_mode else "soft_trim";
        decisions.append(
            EvictionDecision(
                tool_result=tr,
                kind=kind,
                head_chars=self.soft_head_chars,
                tail_chars=self.soft_tail_chars
            )
        );
        # Estimate post-decision visible tokens (placeholder ≈ 8 tokens
        # for hard, head+tail+marker ≈ before for tiny soft trims).
        if kind == "hard_clear" {
            running_tool_tokens -= max(0, before - 8);
        } else {
            cap = self.soft_head_chars + self.soft_tail_chars;
            cur_chars = len(tr.visible_content);
            if cur_chars > cap {
                trimmed_tokens = (cap + 25) // 4;
                running_tool_tokens -= max(0, before - trimmed_tokens);
            }
        }
    }
    return decisions;
}


"""Public context config. Pass to `by llm(context=...)` to opt in.

Holds the live `ContextGraph` after the first call so the caller can
inspect it. With `persist=True` the same graph is reused across
invocations of the decorated function (within the same process). With
`persist=False` a fresh graph is created on every invocation —
discarding the previous one.

Optional `policy` is the strategy interface for pruning/compaction/
materialization. When None (the default), byLLM records into the graph
but doesn't act on it. When set, byLLM dispatches to the policy's hooks
at iteration boundaries, replacing the framework's inline compaction
code path.
"""
obj Context {
    has persist: bool = True,
        session_id: str = "",
        policy: ContextPolicy | None = None,
        # Set by byLLM at call time. Points at the live ContextGraph node.
        graph: ContextGraph | None = None;

    def get_or_create_graph() -> ContextGraph;
    def reset() -> None;
}


impl ContextGraph.mint_tool_result_id(turn: int, tool_name: str) -> str {
    self._tr_seq += 1;
    return f"tr_{turn}_{tool_name}_{self._tr_seq}";
}


impl ContextGraph.mint_message_id(turn: int, role: str) -> str {
    self._msg_seq += 1;
    return f"msg_{turn}_{role}_{self._msg_seq}";
}


impl ContextGraph.record_tool_result(
    tool_name: str,
    args: dict,
    full_content: str,
    visible_content: str,
    turn: int,
    truncation_state: str = "full",
    call_id: str = ""
) -> object {
    tr_id = self.mint_tool_result_id(turn, tool_name);
    tr = ContextToolResult(
        tr_id=tr_id,
        tool_name=tool_name,
        args=args,
        full_content=full_content,
        visible_content=visible_content if visible_content else full_content,
        produced_at=time.time(),
        produced_at_turn=turn,
        truncation_state=truncation_state,
        call_id=call_id
    );
    self ++> tr;
    return tr;
}


impl ContextGraph.record_message(
    role: str,
    turn: int,
    content: str,
    token_estimate: int = 0,
    tool_calls: list | None = None
) -> object {
    msg_id = self.mint_message_id(turn, role);
    msg = ContextMessage(
        msg_id=msg_id,
        role=role,
        turn=turn,
        content=content,
        token_estimate=token_estimate,
        tool_calls=tool_calls if tool_calls is not None else []
    );
    self ++> msg;
    return msg;
}


impl Context.get_or_create_graph() -> ContextGraph {
    if self.graph is not None and self.persist {
        return self.graph;
    }
    new_graph = ContextGraph(session_id=self.session_id, created_at=time.time());
    # Attach to the Jac root so the graph is part of the persistent
    # graph layer. Caller can also reach it via `Context.graph` directly.
    root() ++> new_graph;
    self.graph = new_graph;
    return new_graph;
}


impl Context.reset() -> None {
    self.graph = None;
}
