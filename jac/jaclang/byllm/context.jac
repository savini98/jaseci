import hashlib;
import json;
import math;
import re;
import time;
import from jaclang.byllm.types { Message, Text, ToolCallResultMsg }

glob COMPACTION_TAG: str = "[Compacted context summary]",
     SOFT_TRIM_MARKER: str = "\n...[soft-trimmed]...\n",
     HARD_CLEAR_DEFAULT: str = "[Old tool result content cleared]",
     FOLDED_RESULT_NOTE: str = "[Tool result folded into the compacted context summary]",
     MEDIA_TAG: str = "[media]",
     IDENT_CACHE: dict[str, set[str]] = {},
     LEXTOK_CACHE: dict[str, set[str]] = {},
     STAT_COUNTERS: list[str] = [
         "prunes",
         "restores",
         "trims",
         "stubs",
         "clears",
         "compactions"
     ];

node ContextGraph {
    has session_id: str = "",
        created_at: float = 0.0,
        turn_count: int = 0,
        _tr_seq: int = 0,
        _msg_seq: int = 0,
        citation_counts: dict[str, float] = {},
        citation_turns: dict[str, list[int]] = {},
        citation_idents: dict[str, list[str]] = {},
        n_seen_asst: int = 0,
        trim_memo: dict[str, str] = {},
        append_restored: dict[str, bool] = {},
        stats: dict[str, any] = {};

    def mint_tool_result_id(turn: int, tool_name: str) -> str;
    def mint_message_id(turn: int, role: str) -> str;
    def record_tool_result(
        tool_name: str,
        args: dict[str, any],
        full_content: str,
        turn: int = -1,
        call_id: str = ""
    ) -> ContextToolResult;

    def record_message(
        role: str, content: str, tool_calls: list[dict[str, any]] | None = None
    ) -> ContextMessage;

    def tool_results -> list[ContextToolResult];
    def messages -> list[ContextMessage];
    def summaries -> list[ContextCompactionSummary];
}

node ContextToolResult {
    has tr_id: str,
        tool_name: str,
        full_content: str,
        args: dict[str, any] = {},
        visible_content: str = "",
        produced_at: float = 0.0,
        produced_at_turn: int = 0,
        truncation_state: str = "full",
        visible: bool = True,
        reference_count: int = 0,
        last_referenced_at: float = 0.0,
        call_id: str = "";
}

node ContextMessage {
    has msg_id: str,
        role: str,
        turn: int,
        content: str,
        token_estimate: int = 0,
        visible: bool = True,
        tool_calls: list[dict[str, any]] = [];
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

edge ReferencedBy: ContextMessage --> ContextToolResult {
    has cite_turn: int = 0;
}

obj EvictionDecision {
    has tool_result: ContextToolResult,
        kind: str = "skip",
        head_chars: int = 1500,
        tail_chars: int = 1500,
        placeholder: str = HARD_CLEAR_DEFAULT;
}

obj ContextPolicy {
    def should_prune(cg: ContextGraph, mt_run: object, model: object) -> bool;
    def prune(cg: ContextGraph, mt_run: object, model: object) -> int;
    def compact(
        cg: ContextGraph, mt_run: object, model: object, keep_recent: int
    ) -> bool;

    def materialize(cg: ContextGraph, mt_run: object) -> list[any] | None;
}

obj Context {
    has persist: bool = True,
        session_id: str = "",
        budget_tokens: int = 0,
        budget_ratio: float = 0.5,
        policy: ContextPolicy | None = None,
        `graph: ContextGraph | None = None,
        _resolved_budget: int = 0;

    def postinit -> None;
    def get_or_create_graph -> ContextGraph;
    def reset -> None;
    def stats -> dict[str, any];
}

def apply_eviction_decision(decision: EvictionDecision) -> int;
def estimate_tokens(text: str) -> int;
def distinctive_idents(text: str) -> set[str];
def lexical_tokens(text: str) -> set[str];
def lexical_set_cosine(a: set[str], b: set[str]) -> float;
def cosine(a: list[float], b: list[float]) -> float;
def unit(v: list[float]) -> list[float];
def cached_idents(text: str, cap: int = 20000) -> set[str];
def cached_lextoks(text: str, cap: int = 20000) -> set[str];
def graph_stats(cg: ContextGraph) -> dict[str, any];
def managed_tool_results(cg: ContextGraph) -> list[ContextToolResult];
def message_role(message: object) -> str;
def message_text(message: object) -> str;
def message_tool_calls(message: object) -> list[dict[str, any]];
def tool_call_id_of(message: object) -> str;
def with_content(message: object, content: str) -> object;
def materialize_overlay(
    cg: ContextGraph, mt_run: object, appended: list[str]
) -> list[any] | None;
