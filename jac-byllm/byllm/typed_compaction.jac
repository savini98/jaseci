"""Typed-slot compaction for byLLM memory.

The default AgePolicy collapses older turns into a free-form prose
summary. byLLM-backed prose summarisers drop specific field-level
values during paraphrase — that's what direction-1-results-v2.md
Phase C.3 measured: the proposed-mode 2/2 collapsed to 0/2 when
compaction was on, because values like `keepalive_seconds=47` did
not survive the summary.

This module replaces the free-form output with a fixed-shape record
so the slots can be inspected and re-merged across compaction passes
without paraphrase loss. Agents that need to recall specific numeric
values, configuration keys, or identifiers should pick
TypedSlotCompactionPolicy over the plain AgePolicy.

Layout intentionally separate from `byllm/memory.jac`: the `by llm()`
decorator on `extract_compaction_slots` would otherwise create a
circular import (`memory` → `llm` → `memory`). The base memory layer
stays pure; this module is an opt-in import for agents that want
typed compaction.
"""

import time;
import from byllm.memory {
    MemoryGraph,
    MemoryToolResult,
    MemoryMessage,
    MemoryCompactionSummary,
    AgePolicy
}


"""Structured slots that survive compaction verbatim.

Each slot is a free-text list whose entries the extractor is
instructed to copy from the source content without paraphrase. The
key load-bearing one is `key_observations`: callers writing recall
queries should grep its entries for the identifier they care about.
"""
obj CompactionSlots {
    has files_read: list[str] = [],
        # One entry per important fact. Format convention:
        # 'identifier = value (source)'. The extractor is told to
        # copy numeric values exactly.
        key_observations: list[str] = [],
        commands_run: list[str] = [],
        decisions: list[str] = [],
        unresolved_questions: list[str] = [];
}


"""LLM-backed extraction of typed compaction slots from a list of items
being compacted. The signature is `by llm()` so byLLM populates the
structured return automatically; the typed return makes value
preservation a property of the schema rather than a hope about how the
summariser paraphrases. `previous_slots` lets the policy merge across
multiple compaction passes (anchored update).
"""
def extract_compaction_slots(
    previous_slots: CompactionSlots | None,
    items: list[dict]
) -> CompactionSlots by llm();
sem extract_compaction_slots = """Extract a structured compaction
summary from the items below. items is a list of {turn, role, content}
dicts representing older conversation turns being summarised.

Hard rules — preserve in the output VERBATIM, do not paraphrase:
- Numeric values (e.g. timeouts, ports, counts, thresholds).
- Configuration keys (snake_case or ALL_CAPS).
- File paths, identifiers, error strings.

Slots:
- files_read: every distinct file/path opened.
- key_observations: one entry per important fact, written as
  'identifier = value (source)'. Examples: 'RATE_LIMIT_BURST_CAPACITY
  = 83 (services/api.py)', 'keepalive_seconds = 47 (config/prod.yaml)'.
- commands_run: every distinct shell command (one entry each).
- decisions: any commitments or conclusions reached.
- unresolved_questions: open questions or follow-ups.

If previous_slots is non-empty, merge so the output supersedes it (do
not lose entries that were in previous_slots).""";


"""Render a CompactionSlots into a human-readable block of text. The
slot data is what later turns rely on for recall; the rendered text is
what goes into MemoryCompactionSummary.summary_text. The shape is
deliberately mechanical so a downstream `materialize` hook (planned)
can re-parse it if it ever needs to."""
def render_compaction_slots(slots: CompactionSlots) -> str {
    parts: list[str] = ["[Typed compaction summary]"];
    if slots.files_read {
        parts.append("Files read: " + ", ".join(slots.files_read));
    }
    if slots.key_observations {
        parts.append("Key observations:");
        for obs in slots.key_observations {
            parts.append(f"  - {obs}");
        }
    }
    if slots.commands_run {
        parts.append("Commands run:");
        for cmd in slots.commands_run {
            parts.append(f"  - {cmd}");
        }
    }
    if slots.decisions {
        parts.append("Decisions:");
        for d in slots.decisions {
            parts.append(f"  - {d}");
        }
    }
    if slots.unresolved_questions {
        parts.append("Unresolved:");
        for q in slots.unresolved_questions {
            parts.append(f"  - {q}");
        }
    }
    return "\n".join(parts);
}


"""Typed-slot compaction policy.

Subclasses AgePolicy: inherits the `should_compact` fill-threshold
trigger and the same selection (everything older than
preserve_recent_turns), but replaces the free-form `compact` with an
LLM-backed extraction into CompactionSlots. The rendered slots become
the summary_text of the new MemoryCompactionSummary node. Compacted
originals are hidden as in AgePolicy.

Fail-soft: if the LLM extraction raises, we write a placeholder
summary instead so a single bad call doesn't permanently break
compaction.
"""
obj TypedSlotCompactionPolicy(AgePolicy) {
    override def compact(
        graph: MemoryGraph, mt_run: object
    ) -> MemoryCompactionSummary | None;
}


impl TypedSlotCompactionPolicy.compact(
    graph: MemoryGraph, mt_run: object
) -> MemoryCompactionSummary | None {
    cutoff = graph.turn_count - self.preserve_recent_turns;
    if cutoff <= 0 {
        return None;
    }
    msgs_to_compact = [
        m for m in [graph -->][?:MemoryMessage]
        if m.visible and m.turn <= cutoff
    ];
    trs_to_compact = [
        tr for tr in [graph -->][?:MemoryToolResult]
        if tr.visible and tr.produced_at_turn <= cutoff
    ];
    if not msgs_to_compact and not trs_to_compact {
        return None;
    }

    items: list[dict] = [];
    for m in msgs_to_compact {
        items.append({
            "turn": m.turn,
            "role": m.role,
            "content": m.content
        });
    }
    for tr in trs_to_compact {
        items.append({
            "turn": tr.produced_at_turn,
            "role": f"tool:{tr.tool_name}",
            "content": tr.full_content
        });
    }
    items.sort(key=lambda x: dict : x["turn"]);

    prior_visible = [
        s for s in [graph -->][?:MemoryCompactionSummary] if s.visible
    ];
    prior_node: MemoryCompactionSummary | None = None;
    if prior_visible {
        prior_node = prior_visible[-1];
    }

    # Ask the LLM for structured slots. Fail soft into a placeholder so
    # one bad extraction doesn't tank the rest of the session.
    summary_text: str = "";
    try {
        new_slots = extract_compaction_slots(None, items);
        summary_text = render_compaction_slots(new_slots);
    } except Exception as e {
        summary_text = (
            "[Typed compaction summary — extractor failed: "
            f"{type(e).__name__}: {str(e)[:120]}]"
        );
    }

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
    if prior_node is not None {
        prior_node.visible = False;
    }
    return summary;
}
