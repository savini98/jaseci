import json;
import from jaclang.byllm._optdeps.loguru { logger }
import from jaclang.byllm.mtir { MTRuntime }
import from jaclang.byllm.types { Message, MessageRole }

glob COMPACTION_EXTRACT_INSTRUCTIONS: str = """\
Extract a structured compaction summary from the JSON document in the user \
message. Its "items" field is a list of {turn, role, content} records: older \
conversation turns being summarised. Its "previous_summary" field is the \
rendered summary of turns compacted earlier, or an empty string.

Hard rules: preserve in the output VERBATIM, do not paraphrase:
- Numeric values (e.g. timeouts, ports, counts, thresholds).
- Configuration keys (snake_case or ALL_CAPS).
- File paths, identifiers, error strings.

Slots:
- files_read: every distinct file/path opened.
- key_observations: one entry per important fact, written as \
'identifier = value (source)'. Examples: 'RATE_LIMIT_BURST_CAPACITY = 83 \
(services/api.py)', 'keepalive_seconds = 47 (config/prod.yaml)'.
- commands_run: every distinct shell command (one entry each).
- decisions: any commitments or conclusions reached.
- unresolved_questions: open questions or follow-ups.

If previous_summary is non-empty, merge so the output supersedes it (do not \
lose entries that were in previous_summary).""",
     COMPACTION_FAILED_PREFIX: str = "[Typed compaction summary - extractor failed: ";

obj CompactionSlots {
    has files_read: list[str] = [],
        key_observations: list[str] = [],
        commands_run: list[str] = [],
        decisions: list[str] = [],
        unresolved_questions: list[str] = [];
}

sem CompactionSlots = "Structured compaction summary. Every entry is copied from the source turns without paraphrase.";
sem CompactionSlots.files_read = "Every distinct file or path opened.";
sem CompactionSlots.key_observations = "One entry per important fact, written as 'identifier = value (source)'. Numeric values, configuration keys, paths and identifiers are copied exactly.";
sem CompactionSlots.commands_run = "Every distinct shell command, one entry each.";
sem CompactionSlots.decisions = "Commitments or conclusions reached.";
sem CompactionSlots.unresolved_questions = "Open questions or follow-ups.";

def render_compaction_slots(slots: CompactionSlots) -> str;

def coerce_compaction_slots(value: object) -> CompactionSlots;

def extract_compaction_slots(
    model: object,
    previous_summary: str,
    items: list[dict[str, any]],
    parent_mt_run: object = None
) -> CompactionSlots;

def compaction_summary_text(
    model: object,
    previous_summary: str,
    items: list[dict[str, any]],
    parent_mt_run: object = None
) -> str;
